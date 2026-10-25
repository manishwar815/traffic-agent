// Traffic Agent: 4-way junction lights on an Arduino Uno / Nano (12 LEDs).
//
// The PC sends one line per update at 115200 baud: 4 letters in N E S W order,
// R = red, A = amber, G = green, '-' = lamp off. Example: "GRRR\n" = North green.
// The PC re-sends every 0.5 s. If nothing valid arrives for WATCHDOG_MS, or a
// line asks for two greens, all approaches flash amber (fail-safe).
//
// Wiring: each pin -> 220 ohm resistor -> LED anode, LED cathode -> GND.

const uint8_t PINS[4][3] = {   // {red, amber, green}
  {2, 3, 4},     // N
  {5, 6, 7},     // E
  {8, 9, 10},    // S
  {11, 12, 13},  // W
};
const unsigned long WATCHDOG_MS = 2000;
const unsigned long FLASH_MS = 500;

char line[8];
uint8_t len = 0;
unsigned long lastValid = 0;
bool haveState = false;

void setLamp(uint8_t a, char c) {
  digitalWrite(PINS[a][0], c == 'R');
  digitalWrite(PINS[a][1], c == 'A');
  digitalWrite(PINS[a][2], c == 'G');
}

bool applyLine() {
  if (len != 4) return false;
  uint8_t greens = 0;
  for (uint8_t a = 0; a < 4; a++) {
    char c = line[a];
    if (c != 'R' && c != 'A' && c != 'G' && c != '-') return false;
    if (c == 'G') greens++;
  }
  if (greens > 1) return false;
  for (uint8_t a = 0; a < 4; a++) setLamp(a, line[a]);
  return true;
}

void setup() {
  for (uint8_t a = 0; a < 4; a++)
    for (uint8_t k = 0; k < 3; k++) pinMode(PINS[a][k], OUTPUT);
  Serial.begin(115200);
}

void loop() {
  while (Serial.available()) {
    char c = Serial.read();
    if (c == '\r') continue;
    if (c == '\n') {
      if (applyLine()) {
        lastValid = millis();
        haveState = true;
      }
      len = 0;
    } else if (len < sizeof(line)) {
      line[len++] = c;
    } else {
      len = sizeof(line);  // overlong line: stays invalid until the next '\n'
    }
  }

  if (!haveState || millis() - lastValid > WATCHDOG_MS) {
    haveState = false;
    char c = (millis() / FLASH_MS) % 2 ? 'A' : '-';
    for (uint8_t a = 0; a < 4; a++) setLamp(a, c);
  }
}
