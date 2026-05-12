# serial_comm.py
import serial, time, json

class SerialComm:
    def __init__(self, port="/dev/ttyACM0", baud=115200, timeout=1):  #"/dev/ttyUSB0" "/dev/ttyACM0"
        self.ser = serial.Serial(port, baud, timeout=timeout)
        time.sleep(2)

    def send_json(self, obj):
        s = json.dumps(obj, separators=(',',':')) + '\n'
        #print(f"Enviando al TTGO_J → {s.strip()}") 
        self.ser.write(s.encode())

    def read_json(self):
        try:
            line = self.ser.readline()
            if not line:
                return None
            text = line.decode(errors='ignore').strip()
            if not text:
                return None
            return json.loads(text)
        except Exception:
            return None
