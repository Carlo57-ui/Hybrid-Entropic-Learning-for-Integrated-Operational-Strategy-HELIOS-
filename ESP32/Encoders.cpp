// Encoders.cpp

#include "Encoders.h"

volatile long ticksR = 0;
volatile long ticksL = 0;

const int ENC_R_A = 32;
const int ENC_R_B = 33;

const int ENC_L_A = 34;
const int ENC_L_B = 35;

static float omegaR = 0.0;
static float omegaL = 0.0;

static long prevTicksR = 0;
static long prevTicksL = 0;

static unsigned long prevTime = 0;

const float PPR = 520.0;

void IRAM_ATTR isrRight()
{
    bool A = digitalRead(ENC_R_A);
    bool B = digitalRead(ENC_R_B);

    if(A == B)
        ticksR++;
    else
        ticksR--;
}

void IRAM_ATTR isrLeft()
{
    bool A = digitalRead(ENC_L_A);
    bool B = digitalRead(ENC_L_B);

    if(A == B)
        ticksL++;
    else
        ticksL--;
}

void inicializarEncoders()
{
    pinMode(ENC_R_A, INPUT);
    pinMode(ENC_R_B, INPUT);

    pinMode(ENC_L_A, INPUT);
    pinMode(ENC_L_B, INPUT);

    attachInterrupt(
        digitalPinToInterrupt(ENC_R_A),
        isrRight,
        CHANGE);

    attachInterrupt(
        digitalPinToInterrupt(ENC_L_A),
        isrLeft,
        CHANGE);

    prevTime = millis();
}

void actualizarVelocidades()
{
    unsigned long now = millis();

    float dt = (now - prevTime) / 1000.0;

    if(dt < 0.01)
        return;

    long tR = ticksR;
    long tL = ticksL;

    long dR = tR - prevTicksR;
    long dL = tL - prevTicksL;

    prevTicksR = tR;
    prevTicksL = tL;

    prevTime = now;

    omegaR =
        ((float)dR / PPR) *
        (2.0 * PI) / dt;

    omegaL =
        ((float)dL / PPR) *
        (2.0 * PI) / dt;
}

float getOmegaRight()
{
    return omegaR;
}

float getOmegaLeft()
{
    return omegaL;
}

long getTicksRight()
{
    return ticksR;
}

long getTicksLeft()
{
    return ticksL;
}

void resetEncoders()
{
    ticksR = 0;
    ticksL = 0;
}