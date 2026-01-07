# GPIO Interrupts on KL25Z

Instead of constantly checking (polling) a pin's state, you can use **interrupts** to react only when something changes. This is more efficient and frees up the CPU for other tasks.

---

## 1. Enable Clocks

Enable clocks for both the **PORT** (for interrupt config) and **GPIO** (for pin control):

```c
SIM->SCGC5 |= SIM_SCGC5_PORTA_MASK;  // Port A for button
SIM->SCGC5 |= SIM_SCGC5_PORTB_MASK;  // Port B for LEDs
```

---

## 2. Configure the Interrupt Pin (PCR Register)

Set the **IRQC field** in `PORTx->PCR[n]` to choose when the interrupt fires:

```c
// PTA4 as GPIO input with interrupt on falling edge
PORTA->PCR[4] &= ~PORT_PCR_MUX_MASK;
PORTA->PCR[4] |= PORT_PCR_MUX(1);           // GPIO mode
PORTA->PCR[4] |= PORT_PCR_IRQC(0xA);        // Interrupt on falling edge
```

| IRQC Value | Trigger Condition |
|------------|-------------------|
| `0x0` | Interrupt disabled |
| `0x8` | Interrupt on logic 0 |
| `0x9` | Interrupt on rising edge |
| `0xA` | Interrupt on falling edge |
| `0xB` | Interrupt on either edge |
| `0xC` | Interrupt on logic 1 |

---

## 3. Enable the Interrupt in NVIC

The **NVIC** (Nested Vectored Interrupt Controller) must be told to accept interrupts from your port:

```c
NVIC_EnableIRQ(PORTA_IRQn);  // Enable Port A interrupts
```

Each port has its own IRQ number:

| Port | IRQ Name |
|------|----------|
| Port A | `PORTA_IRQn` |
| Port D | `PORTD_IRQn` |

---

## 4. Write the Interrupt Service Routine (ISR)

The ISR runs automatically when the interrupt fires. **You must clear the interrupt flag** or it will keep firing:

```c
void PORTA_IRQHandler(void) {
    // Your code here...

    // Clear the interrupt flag (MUST DO THIS!)
    PORTA->ISFR = (1 << 4);  // Clear flag for PTA4
}
```

> **Important**: The ISR function name must match exactly (e.g., `PORTA_IRQHandler`).

---

## 5. Keep ISRs Short!

| ✅ Do | ❌ Don't |
|-------|----------|
| Set a flag | Do complex calculations |
| Update a counter | Use long delays |
| Toggle a pin | Call blocking functions |

---

## Summary: Interrupt Setup Steps

1. **Enable clocks** → `SIM->SCGC5`
2. **Configure pin as GPIO** → `PCR` with MUX=1
3. **Set interrupt trigger** → `PCR` with IRQC field
4. **Enable in NVIC** → `NVIC_EnableIRQ()`
5. **Write ISR** → Clear flag at the end!

---

## Example: 2-Bit Counter with Button Interrupt

Press a button to increment a counter displayed on 2 LEDs (counts 0→1→2→3→0...):

```c
volatile int count = 0;

int main(void) {
    /* Enable clocks */
    SIM->SCGC5 |= SIM_SCGC5_PORTA_MASK;  // Button on PTA4
    SIM->SCGC5 |= SIM_SCGC5_PORTB_MASK;  // LEDs on PTB0, PTB1

    /* Configure button PTA4 as input with falling edge interrupt */
    PORTA->PCR[4] &= ~PORT_PCR_MUX_MASK;
    PORTA->PCR[4] |= PORT_PCR_MUX(1);
    PORTA->PCR[4] |= PORT_PCR_IRQC(0xA);  // Falling edge
    PTA->PDDR &= ~MASK(4);                // Input

    /* Configure LEDs PTB0, PTB1 as outputs */
    PORTB->PCR[0] = PORT_PCR_MUX(1);
    PORTB->PCR[1] = PORT_PCR_MUX(1);
    PTB->PDDR |= MASK(0) | MASK(1);       // Outputs
    PTB->PSOR = MASK(0) | MASK(1);        // LEDs OFF initially

    /* Enable interrupt in NVIC */
    NVIC_EnableIRQ(PORTA_IRQn);

    while (1) {
        /* Update LEDs based on count */
        if (count & 0x1) {
            PTB->PCOR = MASK(0);  // LED0 ON
        } else {
            PTB->PSOR = MASK(0);  // LED0 OFF
        }

        if (count & 0x2) {
            PTB->PCOR = MASK(1);  // LED1 ON
        } else {
            PTB->PSOR = MASK(1);  // LED1 OFF
        }
    }
}

/* Button press ISR */
void PORTA_IRQHandler(void) {
    count = (count + 1) & 0x3;   // Increment, wrap at 3

    PORTA->ISFR = (1 << 4);      // Clear interrupt flag
}
```

> **Note**: This example assumes `MASK()` and register definitions already exist. The `volatile` keyword ensures the compiler doesn't optimize away reads of `count` in the main loop.
