import queue
import threading
from scapy.all import sniff
import time

class BackgroundSniffer:
    def __init__(self):
        self.packet_queue = queue.Queue(maxsize=2000)
        self.stop_event = threading.Event()
        self.thread = None
        self.error_msg = None

    def start(self):
        self.packet_queue = queue.Queue(maxsize=2000)
        self.stop_event.clear()
        self.error_msg = None
        self.thread = threading.Thread(target=self._sniff_loop, daemon=True)
        self.thread.start()

    def stop(self):
        self.stop_event.set()
        if self.thread and self.thread.is_alive():
            # Wait briefly for thread to finish
            self.thread.join(timeout=1.0)
        self.thread = None

    def _sniff_loop(self):
        def prn_callback(pkt):
            if self.stop_event.is_set():
                return
            try:
                parsed = self._parse_packet(pkt)
                if parsed:
                    try:
                        self.packet_queue.put(parsed, block=False)
                    except queue.Full:
                        try:
                            self.packet_queue.get_nowait()
                            self.packet_queue.put(parsed, block=False)
                        except queue.Empty:
                            pass
            except Exception as e:
                pass

        try:
            # We run sniff in a loop checking stop_event, with a timeout so it frequently checks stop_event
            while not self.stop_event.is_set():
                sniff(prn=prn_callback, timeout=0.5, store=False)
        except Exception as e:
            self.error_msg = str(e)
            print(f"[!] Sniffing failed: {e}")

    def _parse_packet(self, pkt):
        # We only care about IP packets as Kitsune is an IP-level NIDS
        if not pkt.haslayer('IP'):
            return None
        
        ip = pkt['IP']
        src_ip = ip.src
        dst_ip = ip.dst
        
        src_port = 0
        dst_port = 0
        if pkt.haslayer('TCP'):
            src_port = pkt['TCP'].sport
            dst_port = pkt['TCP'].dport
        elif pkt.haslayer('UDP'):
            src_port = pkt['UDP'].sport
            dst_port = pkt['UDP'].dport
            
        src_mac = "00:00:00:00:00:00"
        if pkt.haslayer('Ether'):
            src_mac = pkt['Ether'].src
            
        size = len(pkt)
        t = getattr(pkt, 'time', None)
        if t is None:
            t = time.time()
        else:
            t = float(t)
            
        return (t, src_mac, src_ip, src_port, dst_ip, dst_port, size)
