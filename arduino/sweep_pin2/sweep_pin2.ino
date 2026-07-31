// Sweep one PDI 6225MG-300 servo on pin 2 through its full ~300° range and back.
// Uses microsecond pulse widths (500–2500 µs) so we hit the true mechanical ends.

#include <Servo.h>

const int SERVO_PIN = 2;
const int MIN_US    = 500;     // ~0°   mechanical
const int MAX_US    = 2500;    // ~300° mechanical
const int STEP_US   = 10;      // smaller = smoother + slower
const int STEP_DELAY_MS = 15;  // delay between micro-steps

Servo servo;

void setup() {
  Serial.begin(115200);
  servo.attach(SERVO_PIN, MIN_US, MAX_US);
  servo.writeMicroseconds(MIN_US);
  delay(500);
}

void loop() {
  // forward sweep: 0° → 300°
  for (int us = MIN_US; us <= MAX_US; us += STEP_US) {
    servo.writeMicroseconds(us);
    delay(STEP_DELAY_MS);
  }
  delay(400);

  // reverse sweep: 300° → 0°
  for (int us = MAX_US; us >= MIN_US; us -= STEP_US) {
    servo.writeMicroseconds(us);
    delay(STEP_DELAY_MS);
  }
  delay(400);
}
