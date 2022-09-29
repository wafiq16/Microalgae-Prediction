#include <Wire.h>
#include "Adafruit_TCS34725.h"

#define LED_PINW    5 //pin led bawah
#define LED_PIN     6 //pin led sensor
#define NUM_LEDS    20 //jumlah led

int R = 0;
int G = 0;
int B = 0;

float rawLowR=0;
float rawLowG=0;
float rawLowB=0;
float rawHighR=0;
float rawHighG=0;
float rawHighB=0;

float referanceLowR=0;
float referanceLowG=0;
float referanceLowB=0;
float referanceHighR=256;
float referanceHighG=256;
float referanceHighB=256;

float referanceRangeR=referanceHighR-referanceLowR;
float referanceRangeG=referanceHighG-referanceLowG;
float referanceRangeB=referanceHighB-referanceLowB;

float correctedValueR;
float correctedValueG;
float correctedValueB;

float rawRangeR;
float rawRangeG;
float rawRangeB;

Adafruit_TCS34725 tcs = Adafruit_TCS34725(TCS34725_INTEGRATIONTIME_700MS, TCS34725_GAIN_1X); //inisialisasi tcs34725

void setup() {
  // put your setup code here, to run once:
  Serial.begin(9600);
  tcs.begin();
  pinMode(LED_PIN, OUTPUT);
  pinMode(LED_PINW, OUTPUT);
  digitalWrite(LED_PINW, LOW);
  digitalWrite(LED_PIN, LOW);
}

void loop()
{
  // put your main code here, to run repeatedly:
  uint16_t red, green, blue, clr, colorTemp, lux;
  tcs.getRawData(&red, &green, &blue, &clr);
  colorTemp = tcs.calculateColorTemperature_dn40(red, green, blue, clr);
  lux = tcs.calculateLux(red, green, blue);

  uint32_t sum = clr;
  float r, g, b;

  r = red;
  r /= sum;

  g = green;
  g /= sum;

  b = blue;
  b /= sum;

  r *= 256;
  g *= 256;
  b *= 256;

  R = r;
  G = g;
  B = b;

  if (Serial.available() > 0)
  {
    char req = (char)Serial.read();
    if (req == 'a')
    {
      tcs.getRawData(&red, &green, &blue, &clr);
      Serial.print("@,");
      Serial.print(colorTemp, DEC); Serial.print(",");
      Serial.print(lux, DEC); Serial.print(",");
      Serial.print(red, DEC); Serial.print(",");
      Serial.print(green, DEC); Serial.print(",");
      Serial.print(blue, DEC); Serial.print(",");
      Serial.print(clr, DEC); Serial.println(",$");
    }

    else if (req == 'c')
    {
      rawRangeR=rawHighR-rawLowR;
      rawRangeG=rawHighG-rawLowG;
      rawRangeB=rawHighB-rawLowB;
      correctedValueR=(((R-rawLowR)*referanceRangeR)/rawRangeR)+referanceLowR;
      correctedValueG=(((G-rawLowG)*referanceRangeG)/rawRangeG)+referanceLowG;
      correctedValueB=(((B-rawLowB)*referanceRangeB)/rawRangeB)+referanceLowB;
      Serial.print("@,");
      //Serial.print(colorTemp, DEC); Serial.print(",");
      //Serial.print(lux, DEC); Serial.print(",");
      Serial.print((int)correctedValueR, DEC); Serial.print(",");
      Serial.print((int)correctedValueG, DEC); Serial.print(",");
      Serial.print((int)correctedValueB, DEC); Serial.println(",$"); //Serial.print(",");
      //Serial.print(clr, DEC); 
    }
  
    else if (req == 'r')
    {
      
    }
    
    else if (req == 'g')
    {
      
    }

    else if (req == 'b')
    {
      
    }

    else if (req == 'w')
    {
      digitalWrite(LED_PINW, HIGH);
    }

    else if (req == 'm')
    {
      digitalWrite(LED_PINW, LOW);
    }

    else if (req == 'l')
    {
      digitalWrite(LED_PIN, HIGH);
    }

    else if (req == 'n')
    {
      digitalWrite(LED_PIN, LOW);
    }

    else if (req == 'z')
    {
      rawLowR=R;
      rawLowG=G;
      rawLowB=B;
    }

    else if (req == 'q')
    {
      rawHighR=R;
      rawHighG=G;
      rawHighB=B;
    }
  }
}
