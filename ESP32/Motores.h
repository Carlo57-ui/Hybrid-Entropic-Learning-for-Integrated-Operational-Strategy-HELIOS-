// Motores.h

#ifndef MOTORES_H 
#define MOTORES_H 
#include <Arduino.h> 

extern const int ENA; 
extern const int ENB; 
extern const int IN1; 
extern const int IN2; 
extern const int IN3; 
extern const int IN4; 

void inicializarMotores(); 
void aplicarPWM(int pwmR, int pwmL); 
void detenerMotores(); 

#endif