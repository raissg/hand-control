// Per-finger hand controller. 5 fingers + 1 extra servo (pin 11).
// Calibrated for PDI-6225MG (300 deg range, 500..2500 us pulse width).
//
// Bit semantics:
//   bit 0 -> REST_ANGLE[i]   (relaxed / extended -- finger open)
//   bit 1 -> CLOSED_ANGLE[i] (fully flexed / curled)
//
// Protocol (angles are physical degrees, 0..300):
//   F<b0><b1><b2><b3><b4>   per-finger bits, order = thumb,index,middle,ring,pinky.
//                           Each servo locks for servoLockMs after a change
//                           (0 = no lock; see L command).
//   L<dddd>                 set per-finger lock duration in ms, 4 ASCII digits
//                           0000..9999. 0000 disables lock on F commands.
//   A<i><3 digits>          write raw angle 0..300 to servo i (0..5).
//                           Index 5 = pin 11 extra servo.
//                           No lockout. Calibration / manual control.
//                           e.g. "A5090" = pin 11 to 90 deg.
//   D                       park all servos at DEFAULT_ANGLE (keeps torque).
//   S                       stop -- detach all servos (no torque).
// Any F, A, or D after S re-attaches.

#include <Servo.h>

const int NUM_FINGERS = 5;        // F<bits> only addresses these
const int NUM_SERVOS  = 6;        // total servos including pin 11 extra
const int servoPins[NUM_SERVOS] = {3, 5, 6, 9, 10, 11};

// PDI-6225MG: 300 deg, 500..2500 us pulse width.
const int   SERVO_MAX_DEG = 300;
const int   SERVO_US_MIN  = 500;
const int   SERVO_US_MAX  = 2500;

// Per-finger calibrated angles (physical degrees, measured via slider UI).
// Order: thumb, index, middle, ring, pinky, extra(pin 11).
// Pin 11's REST/CLOSED are placeholders -- tune them with the slider UI.
// REST_ANGLE   = [ 95, 148, 124, 126, 126]

const int REST_ANGLE[NUM_SERVOS]   = { 95, 148, 124, 126, 126, 100};  // bit 0
const int DEFAULT_ANGLE[NUM_SERVOS] = {100,  90,  90,  90,  90, 100};  // park on exit
const int CLOSED_ANGLE[NUM_SERVOS] = { 37,   0, 250, 260, 260, 100};  // bit 1
// pin 11 (index 5): both endpoints set to 100 -- it never moves from there.

// Default 2000 ms; Python can override at runtime with L2000 / L0000 etc.
unsigned long servoLockMs = 2000;

Servo servos[NUM_SERVOS];
int currentBit[NUM_FINGERS] = {0, 0, 0, 0, 0};
unsigned long lastChange[NUM_FINGERS] = {0, 0, 0, 0, 0};
bool attached = false;

int angleToUs(int deg) {
  if (deg < 0) deg = 0;
  if (deg > SERVO_MAX_DEG) deg = SERVO_MAX_DEG;
  long span = (long)(SERVO_US_MAX - SERVO_US_MIN);
  return (int)(SERVO_US_MIN + (long)deg * span / SERVO_MAX_DEG);
}

void attachAll() {
  for (int i = 0; i < NUM_SERVOS; i++) {
    servos[i].attach(servoPins[i], SERVO_US_MIN, SERVO_US_MAX);
  }
  attached = true;
}

void detachAll() {
  for (int i = 0; i < NUM_SERVOS; i++) servos[i].detach();
  attached = false;
}

void writeFinger(int i, int bit) {
  int angle = bit ? CLOSED_ANGLE[i] : REST_ANGLE[i];
  servos[i].writeMicroseconds(angleToUs(angle));
}

void printState(char cmd) {
  Serial.print("cmd=");
  Serial.print(cmd);
  Serial.print("  bits=");
  for (int i = 0; i < NUM_FINGERS; i++) Serial.print(currentBit[i]);
  Serial.println();
}

void writeDefaultAngles() {
  if (!attached) attachAll();
  for (int i = 0; i < NUM_SERVOS; i++) {
    servos[i].writeMicroseconds(angleToUs(DEFAULT_ANGLE[i]));
  }
  for (int i = 0; i < NUM_FINGERS; i++) currentBit[i] = 0;
  Serial.print("cmd=D  default=");
  for (int i = 0; i < NUM_SERVOS; i++) {
    if (i) Serial.print(',');
    Serial.print(DEFAULT_ANGLE[i]);
  }
  Serial.println();
}

void setup() {
  Serial.begin(115200);
  // Cap readBytes() stalls at 20 ms so a partial command can't freeze loop()
  // for the default 1 s while the host's RX buffer overflows.
  Serial.setTimeout(20);
  attachAll();
  unsigned long now = millis();
  for (int i = 0; i < NUM_FINGERS; i++) {
    currentBit[i] = 0;             // start at rest (open hand) -- bit 0 = REST
    lastChange[i] = (servoLockMs > 0) ? (now - servoLockMs) : 0UL; // first F not blocked
    writeFinger(i, 0);
  }
  // Pin 11 (extra) is parked at 100 deg and never moved by F/A handlers below.
  servos[5].writeMicroseconds(angleToUs(100));
  Serial.print("READY  6 servos on pins {3,5,6,9,10,11}  range=0..");
  Serial.print(SERVO_MAX_DEG);
  Serial.println("deg");
  for (int i = 0; i < NUM_SERVOS; i++) {
    Serial.print(i == 5 ? "  extra  " : "  finger "); Serial.print(i);
    Serial.print(": rest="); Serial.print(REST_ANGLE[i]);
    Serial.print(" closed="); Serial.println(CLOSED_ANGLE[i]);
  }
  Serial.print("  lock="); Serial.print(servoLockMs);
  Serial.println("ms  (Ldddd to change, 0000=off)");
}

void loop() {
  if (Serial.available() <= 0) return;

  char cmd = Serial.read();

  if (cmd == 'D' || cmd == 'd') {
    writeDefaultAngles();
    return;
  }

  if (cmd == 'S' || cmd == 's') {
    detachAll();
    Serial.println("cmd=S  detached");
    return;
  }

  if (cmd == 'L' || cmd == 'l') {
    // L#### : set servo lock duration in ms (0000 = no lock on F handler).
    char b[4];
    if (Serial.readBytes(b, 4) != 4) return;
    for (int k = 0; k < 4; k++) {
      if (b[k] < '0' || b[k] > '9') return;
    }
    unsigned long v = (b[0] - '0') * 1000UL + (b[1] - '0') * 100UL
                    + (b[2] - '0') * 10UL + (b[3] - '0');
    servoLockMs = v;
    unsigned long nowL = millis();
    for (int i = 0; i < NUM_FINGERS; i++) {
      lastChange[i] = (servoLockMs > 0) ? (nowL - servoLockMs) : 0UL;
    }
    Serial.print("cmd=L  lock="); Serial.print(servoLockMs);
    Serial.println("ms");
    return;
  }

  if (cmd == 'A' || cmd == 'a') {
    // A<i><3 digits>: raw angle in 0..SERVO_MAX_DEG, bypass lockout.
    char buf[4];
    int n = Serial.readBytes(buf, 4);
    if (n != 4) return;
    int i = buf[0] - '0';
    if (i < 0 || i >= NUM_SERVOS) return;   // 0..4 fingers, 5 = extra (pin 11)
    if (buf[1] < '0' || buf[1] > '9' ||
        buf[2] < '0' || buf[2] > '9' ||
        buf[3] < '0' || buf[3] > '9') return;
    int a = (buf[1]-'0')*100 + (buf[2]-'0')*10 + (buf[3]-'0');
    if (a < 0 || a > SERVO_MAX_DEG) return;
    if (!attached) attachAll();
    servos[i].writeMicroseconds(angleToUs(a));
    // No echo for A: at ~300 cmds/s the per-command print saturates the
    // serial link, fills the RX buffer, drops bytes, and triggers
    // readBytes() timeouts that freeze the loop for seconds.
    return;
  }

  if (cmd == 'F' || cmd == 'f') {
    char buf[NUM_FINGERS];
    int n = Serial.readBytes(buf, NUM_FINGERS);
    if (n != NUM_FINGERS) return;
    if (!attached) attachAll();
    unsigned long now = millis();
    bool changed = false;
    for (int i = 0; i < NUM_FINGERS; i++) {
      if (buf[i] != '0' && buf[i] != '1') continue;
      int newBit = buf[i] - '0';
      if (newBit == currentBit[i]) continue;
      if (servoLockMs > 0 && (unsigned long)(now - lastChange[i]) < servoLockMs) continue;
      currentBit[i] = newBit;
      lastChange[i] = now;
      writeFinger(i, newBit);
      changed = true;
    }
    if (changed) printState('F');
  }
  // unknown bytes (newline, etc.) silently ignored
}
