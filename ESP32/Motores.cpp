// Encoders.cpp

#include "Motores.h" 
const int IN1 = 2; 
const int IN2 = 15; 
const int IN3 = 5; 
const int IN4 = 18; 
const int ENA = 19; 
const int ENB = 21; 

const int PWM_FREQ = 20000; 
const int PWM_RES = 8; 

void inicializarMotores() 
{ 
    pinMode(IN1, OUTPUT); 
    pinMode(IN2, OUTPUT); 
    pinMode(IN3, OUTPUT); 
    pinMode(IN4, OUTPUT); 
    
    ledcAttach(ENA, PWM_FREQ, PWM_RES); 
    ledcAttach(ENB, PWM_FREQ, PWM_RES); 
    detenerMotores(); 
} 

void aplicarPWM(int pwmR, int pwmL) 
{ 
    pwmR = constrain(pwmR, -255, 255); 
    pwmL = constrain(pwmL, -255, 255); 
    
    if (pwmL >= 0) 
    { 
        digitalWrite(IN1, LOW); 
        digitalWrite(IN2, HIGH); 
    } 
    else 
    { 
        digitalWrite(IN1, HIGH); 
        digitalWrite(IN2, LOW);
    } 
    
    if (pwmR >= 0) 
    { 
        digitalWrite(IN3, HIGH); 
        digitalWrite(IN4, LOW); 
    } 
    else 
    { 
        digitalWrite(IN3, LOW); 
        digitalWrite(IN4, HIGH); 
    } 
    
    ledcWrite(ENA, abs(pwmR)); 
    ledcWrite(ENB, abs(pwmL)); 
} 

void detenerMotores() 
{ 
    ledcWrite(ENA, 0); 
    ledcWrite(ENB, 0); 
    
    digitalWrite(IN1, LOW); 
    digitalWrite(IN2, LOW); 
    digitalWrite(IN3, LOW); 
    digitalWrite(IN4, LOW); 
}