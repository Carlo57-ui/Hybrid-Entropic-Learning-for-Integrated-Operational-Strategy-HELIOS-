// Encoders.h

#ifndef ENCODERS_H
#define ENCODERS_H

#include <Arduino.h>

void inicializarEncoders();

long getTicksRight();
long getTicksLeft();

void resetEncoders();

void actualizarVelocidades();

float getOmegaRight();
float getOmegaLeft();

#endif