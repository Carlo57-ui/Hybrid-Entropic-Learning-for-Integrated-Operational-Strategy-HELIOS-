// ESP32.ino

#include <Arduino.h>
#include <ArduinoJson.h>

#include "Motores.h"
#include "Encoders.h"


const float R = 0.05;      // radio rueda [m]
const float L = 0.1559;    // separación ruedas [m]

// Referencias recibidas desde Python
float v_ref = 0.0;
float w_ref = 0.0;

// Control PI

float Kp_R = 3.0;
float Ki_R = 0.5;

float Kp_L = 3.0;
float Ki_L = 0.5;

float intR = 0.0;
float intL = 0.0;


unsigned long lastControl = 0;
const float Ts = 0.05;


void setup()
{
    Serial.begin(115200);

    inicializarMotores();
    inicializarEncoders();

    Serial.println("Control PI iniciado");
}


void loop()
{

    // Leer referencias desde Python

    if (Serial.available())
    {
        String line = Serial.readStringUntil('\n');
        line.trim();

        int sep = line.indexOf(',');

        if (sep > 0)
        {
            v_ref =
                line.substring(0, sep).toFloat();

            w_ref =
                line.substring(sep + 1).toFloat();
        }
    }

    // Ejecutar control cada 50 ms

    if (millis() - lastControl >= 50)
    {
        lastControl = millis();

        actualizarVelocidades();

        // Velocidades medidas

        float omegaR = getOmegaRight();
        float omegaL = getOmegaLeft();

        // Cinemática inversa
        float omegaR_ref =
            (2.0 * v_ref + L * w_ref) /
            (2.0 * R);

        float omegaL_ref =
            (2.0 * v_ref - L * w_ref) /
            (2.0 * R);

        // Error

        float eR =
            omegaR_ref - omegaR;

        float eL =
            omegaL_ref - omegaL;

        // Integrador


        intR += eR * Ts;
        intL += eL * Ts;

        intR = constrain(intR, -20.0f, 20.0f);
        intL = constrain(intL, -20.0f, 20.0f);

        // PI

        float uR =
            Kp_R * eR +
            Ki_R * intR;

        float uL =
            Kp_L * eL +
            Ki_L * intL;

        // PWM calculado

        float pwmR = 0;
        float pwmL = 0;

        // Motor derecho

        if (fabs(omegaR_ref) > 0.05)
        {
            pwmR = 130 + fabs(uR);

            if (omegaR_ref < 0)
                pwmR = -pwmR;
        }

        // Motor izquierdo

        if (fabs(omegaL_ref) > 0.05)
        {
            pwmL = 150 + fabs(uL);

            if (omegaL_ref < 0)
                pwmL = -pwmL;
        }

        // Saturación
        pwmR =
            constrain(pwmR, -255.0f, 255.0f);

        pwmL =
            constrain(pwmL, -255.0f, 255.0f);

        // Aplicar PWM
        aplicarPWM(pwmR, pwmL);

        // Velocidades reales del robot
        float v_real =
            R * (omegaR + omegaL) / 2.0;

        float w_real =
            R * (omegaR - omegaL) / L;


        // Telemetría JSON
        StaticJsonDocument<256> doc;

        doc["v_ref"] = v_ref;
        doc["w_ref"] = w_ref;

        doc["v_real"] = v_real;
        doc["w_real"] = w_real;

        doc["omegaR"] = omegaR;
        doc["omegaL"] = omegaL;

        doc["omegaR_ref"] = omegaR_ref;
        doc["omegaL_ref"] = omegaL_ref;

        doc["eR"] = eR;
        doc["eL"] = eL;

        doc["pwmR"] = pwmR;
        doc["pwmL"] = pwmL;

        serializeJson(doc, Serial);
        Serial.println();
    }
}