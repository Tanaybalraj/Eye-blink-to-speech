# Eye-blink-to-speech
Motor Neuron Disease (MND) is a medical condition where the motor neurons of the patient are paralyzed, it is incurable. It also leads to weakness of muscles with respect to hand, feet or voice. Because of this, the patient cannot perform his voluntary actions and it is very difficult for the patient to express his needs as he is not able to communicate with the world. There are many methods introduced for the motor neuron disease patients to communicate with the outside world such as Brain wave technique and Electro-oculography. Loss of speech can be hard to adjust. It is difficult for the patients to make the caretaker understand what they need especially when they are in hospitals. It becomes difficult for the patients to express their feelings and even they cannot take part in conversations. System incorporates different visual technologies, such as eye blink detection, eye centre localization and conversion of the eye blink to speech. The proposed system detects the eye blink and differentiates between an intentional long blink and a normal eye blink. The proposed system can be used to control and Communicate with other people. The objectives of the system are: Capturing the frame from the video using the system’s camera initialises the execution of the proposed system.The Face Detection Algorithm then processes on the captured video frames to give out the rectangular boxed face. This output from Face Detection Algorithm then gets processed using AdaBoost Classifier to detect the eye region in the face.Eye detected will be sent to check if there is any movement of the eyeball. If it’s there, then this movement will be tracked to give out the combination the patient is using to express the dialogue.If not, then the blink pattern will be processed to give out the voice as well as the text input with respective dialogue.
