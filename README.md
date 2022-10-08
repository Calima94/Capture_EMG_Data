# Capture_EMG_Data

## Sobre o projeto

Este projeto foi desenvolvido de forma a facilitar a captura de categorias pré-definidas do ângulo de cotovelo pelo usuário, utilizando o dispositivo
"Myo-armband" desenvolvido pela empresa Thalmic Labs, conjuntamente com uma webcam de computador/notebook comum.

Agradecimento especial à dzhu que com seu projeto ajudou na comunicação com o dispositivo "Myo":
https://github.com/dzhu/myo-raw

Agradecimento especial também a Alvipe devido as implementações feitas adicionais ao projeto de dzhu, que foram incorporadas ao projeto:
https://github.com/Alvipe/myo-raw

E finalmente a Fernando Cossentino que utilizando a partir do código de dzhu fez modificações que foram utilizadas neste projeto.
http://www.fernandocosentino.net/pyoconnect/ (acessado em 20/06/2022)

Para acessar a versão base deste projeto sem o módulo de leitura de sinais EMG acesse o github de Alan Mendes (alans96):
https://github.com/alans96/arm_robotics

## Modo de Utilização

Execute o arquivo capture_myo_filtered_signal_50hz.py, para capturar os sinais filtrados pelo aparelho Myo a uma taxa de 50Hz, ou execute o arquivo   usando o bracelete "Myo" com o adaptador "bluetooth" conectado em uma porta USB de seu computador.

Na aba "Funções" selecione o braço em que se está realizando as capturas dos sinais EMG.

Em seguida digite a quantidade de amostras por categoria que você deseja capturar (O "Myo" faz aproximadamente 200 capturas por segundo)

Logo após selecione o número de categorias que você deseja classificar, conjuntamente com a variância de ângulo que será aceito para um ângulo

ser pertencente a determinada categoria. (Evite o uso de variâncias que sobrepunham as categorias, elas podem causar erros nas classificações futuras)* ** ***.

Click no botão Start.

Quando o ângulo estiver sendo capturado as linhas do braço do utilizador ficará laranja.

Após ser capturado uma das categorias com a quantidade de amostras especificadas elas apareceram em verde.

Após todas as categorias serem treinadas a janela de exibição da captura das amostras se encerará e o programa salvará automaticamente.

*Selecionando 2 categorias o algoritmo identificará (180° como categoria 1 e 90° como categoria 2)

**Selecionando 3 categorias o algoritmo identificará (180° como categoria 1 e 90° como categoria 2, 60° como categoria 3)

***Selecionando 4 categorias o algoritmo identificará (180° como categoria 1 e 90° como categoria 2, 60° como categoria 3, 45° como categoria 4)


## Requisitos e cuidados na execução do programa

Este sistema foi testado para sistemas Linux (Especificamente o Ubuntu 20.04)

Para a correta utilização deste programa veja os requisitos do projeto descrito no arquivo "requirements.txt"

Caso haja problemas com o arquivo o PyQt5, especificamente com o arquivo "xbc", pode ser problemas de compatibilidade do OpenCv com o PyQt5, desta forma
procure o artigo no "StackOverflow" abaixo pode ajudar.

https://stackoverflow.com/questions/63829991/qt-qpa-plugin-could-not-load-the-qt-platform-plugin-xcb-in-even-though-it

Utilizando o comando:
(pip3 install PyQt5==5.11.3)

Caso também haja problemas com a importação do arquivo cv2, o artigo do "StackOverflow" também pode ajudar.

https://stackoverflow.com/questions/37776228/pycharm-python-opencv-and-cv2-install-error

!pip install opencv-contrib-python    #working on both Windows and Ubuntu

