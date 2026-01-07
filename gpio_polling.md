# GPIO Configuration on KL25Z

GPIO (General Purpose Input/Output) pins let you control LEDs, read buttons, and interface with external hardware. Here's what you need to configure.

---

## 1. Enable the Clock

Before using any GPIO port, enable its clock in **SIM_SCGC5**:

```c
SIM->SCGC5 |= SIM_SCGC5_PORTA_MASK;  // Enable clock for Port A
```

| Port | Mask |
|------|------|
| Port A | `SIM_SCGC5_PORTA_MASK` |
| Port B | `SIM_SCGC5_PORTB_MASK` |
| Port C | `SIM_SCGC5_PORTC_MASK` |
| Port D | `SIM_SCGC5_PORTD_MASK` |
| Port E | `SIM_SCGC5_PORTE_MASK` |

---

## 2. Configure Pin Mux (PCR Register)

Each pin can serve multiple functions (GPIO, UART, SPI, etc.). The **MUX field** in `PORTx->PCR[n]` selects the function:

```c
// Set PTA0 to GPIO mode (MUX = 1)
PORTA->PCR[0] &= ~PORT_PCR_MUX_MASK;  // Clear MUX bits
PORTA->PCR[0] |= PORT_PCR_MUX(1);     // Set MUX to 1 (GPIO)
```

| MUX Value | Function |
|-----------|----------|
| 0 | Disabled (analog or default) |
| 1 | **GPIO** |
| 2–7 | Alternate functions (UART, SPI, etc.) |

---

## 3. Set Pin Direction (PDDR Register)

Use **GPIOx->PDDR** to set pins as input or output:

```c
// Set PTA0, PTA1, PTA2 as outputs
PTA->PDDR |= MASK(0) | MASK(1) | MASK(2);
```

| Bit Value | Direction |
|-----------|-----------|
| 0 | Input |
| 1 | Output |

---

## 4. Control Output Pins

Use these registers to control output values:

| Register | Action | Example |
|----------|--------|---------|
| **PSOR** | Set pin HIGH | `PTA->PSOR = MASK(0);` |
| **PCOR** | Clear pin LOW | `PTA->PCOR = MASK(0);` |
| **PTOR** | Toggle pin | `PTA->PTOR = MASK(0);` |
| **PDOR** | Write all pins | `PTA->PDOR = 0x07;` |

---

## 5. Read Input Pins

Use **PDIR** to read the current state of input pins:

```c
if (PTA->PDIR & MASK(3)) {
    // PTA3 is HIGH
}
```

---

## Summary: GPIO Setup Steps

1. **Enable clock** → `SIM->SCGC5`
2. **Set pin mux to GPIO** → `PORTx->PCR[n]` with MUX=1
3. **Set direction** → `GPIOx->PDDR`
4. **Control/read pins** → `PSOR`, `PCOR`, `PTOR`, `PDIR`

---

## Example: LED Binary Counter

This code blinks 3 LEDs on PTA0, PTA1, PTA2 in a binary counting pattern (0–7):

```c
int main() { // Write the initial GPIO configuration
    int count = 0;

    /* Enable clock for PORTA */
    SIM->SCGC5 |= SIM_SCGC5_PORTA_MASK;

    /* Configure PTA0, PTA1, PTA2 as GPIO */
    PORTA->PCR[0] &= ~PORT_PCR_MUX_MASK;
    PORTA->PCR[0] |= PORT_PCR_MUX(1);

    PORTA->PCR[1] &= ~PORT_PCR_MUX_MASK;
    PORTA->PCR[1] |= PORT_PCR_MUX(1);

    PORTA->PCR[2] &= ~PORT_PCR_MUX_MASK;
    PORTA->PCR[2] |= PORT_PCR_MUX(1);

    /* Set PTA0, PTA1, PTA2 as outputs */
    PTA->PDDR |= MASK(0) | MASK(1) | MASK(2);

    /* Turn all LEDs OFF initially (based on the circuit in the figure) */
    PTA->PSOR = MASK(0) | MASK(1) | MASK(2);

    while (1) { // Write the loop
        /* First turn all LEDs OFF */
        PTA->PSOR = MASK(0) | MASK(1) | MASK(2);

        /* Explicitly turn ON LEDs based on count bits */
        if (count & 0x1) {
            PTA->PCOR = MASK(0);   // PTA0 ON
        }
        if (count & 0x2) {
            PTA->PCOR = MASK(1);   // PTA1 ON
        }
        if (count & 0x4) {
            PTA->PCOR = MASK(2);   // PTA2 ON
        }

        Delay(1000);
        count = (count + 1) & 0x7;
    }
}
```

> **Note**: This example assumes `MASK()`, `Delay()`, and register definitions already exist.
