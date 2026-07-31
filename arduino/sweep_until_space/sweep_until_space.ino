// Slowly sweep a PDI 6225MG-300 servo on pin 3 in one direction.
// Press SPACE in the serial monitor to stop. Live angle printed continuously.

#include <Servo.h>

const int SERVO_PIN = 3;
const int MIN_US    = 500;     // ~0°
const int MAX_US    = 2500;    // ~300°
const int START_US  = 1100;    // ~90°  (500 + 90/300 * (2500-500))
const int STEP_US   = 4;       // small step = slow + smooth
const int STEP_DELAY_MS = 40;  // ~25 steps/sec
const int PRINT_EVERY_MS = 100;

Servo servo;
int   us = START_US;
bool  stopped = false;
unsigned long lastPrint = 0;

float usToDegrees(int pulseUs) {
  // Linear map: 500us -> 0deg, 2500us -> 300deg
  return (float)(pulseUs - MIN_US) * 300.0f / (float)(MAX_US - MIN_US);
}

void setup() {
  Serial.begin(115200);
  servo.attach(SERVO_PIN, MIN_US, MAX_US);
  servo.writeMicroseconds(us);
  delay(300);
  Serial.println("Starting at 90 deg. Sweeping pin 3 toward 300 deg. Press SPACE to stop.");
}

void loop() {
  // ---- read serial: any space stops the sweep ----
  while (Serial.available() > 0) {
    char c = Serial.read();
    if (c == ' ') {
      stopped = true;
      Serial.print("STOPPED at ");
      Serial.print(usToDegrees(us), 1);
      Serial.print(" deg  (pulse ");
      Serial.print(us);
      Serial.println(" us)");
    }
  }

  // ---- advance the sweep if not stopped and not at end ----
  if (!stopped && us < MAX_US) {
    us += STEP_US;
    if (us > MAX_US) us = MAX_US;
    servo.writeMicroseconds(us);
    delay(STEP_DELAY_MS);
  }

  // ---- print current angle ~10x/sec ----
  unsigned long now = millis();
  if (now - lastPrint >= PRINT_EVERY_MS) {
    lastPrint = now;
    Serial.print(stopped ? "[STOP] " : "[MOVE] ");
    Serial.print("angle = ");
    Serial.print(usToDegrees(us), 1);
    Serial.print(" deg   pulse = ");
    Serial.print(us);
    Serial.println(" us");
  }
}
