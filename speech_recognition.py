import speech_recognition as sr
import pyttsx3

recognizer = sr.Recognizer()
while True:
    try:
        with sr.Microphone() as mic:
            recognizer.adjust_for_ambient_noise(mic,duration=0.2)
            audio=recognizer.listen(mic)
            text=recognizer.recognize_google(audio,language="fr-FR")  
            text= text.lower()
            print(text)
            n=int(input("entrer 0 pour fermer ou 1 pour continuer :   "))
            if n==0:
               break
    except: 
        print("il y un erreur")
        recognizer = sr.Recognizer()
        continue
print("merci pour ton speech")