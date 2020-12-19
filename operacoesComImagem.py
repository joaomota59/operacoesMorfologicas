#pip install cv2 + skimage + mahotas + numpy + matplotlib
import cv2
from skimage import morphology
from skimage.filters import threshold_otsu
from skimage.util import invert
import mahotas as mh 
import numpy as np
import matplotlib.pyplot as plt
from pylab import imshow,show

def imgOriginal(img):#desenha a imagem original na tela
    try:
        axs[0].set_title('Imagem Original')
        axs[0].set_axis_off()#tira o eixo x e y da imagem que fica na coluna 0
        axs[0].imshow(img,cmap='gray')
    except:
        axs[0][0].set_title('Imagem Original')
        axs[0][0].set_axis_off()#tira o eixo x e y da imagem que fica na coluna 0 linha 0
        axs[0][0].imshow(img,cmap='gray')

def greyToRGB(img):#transforma a imagem de cinza para rgb com as cores originais
    escalaRGB = cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)#transforma a matriz cinza em RGB-3D
    linha,coluna,profundidade = escalaRGB.shape
    for i in range(linha):
      for j in range(coluna):
        if (escalaRGB[i,j]==[115,115,115]).all():#seta os pixels do primeiro desenho para verde
            escalaRGB[i][j][0] = 35
            escalaRGB[i][j][1] = 166
            escalaRGB[i][j][2] = 59
        elif (escalaRGB[i,j]==[88,88,88]).all():#pixels para magenta
            escalaRGB[i][j][0] = 144
            escalaRGB[i][j][1] = 48
            escalaRGB[i][j][2] = 147
        elif (escalaRGB[i,j]==[72,72,72]).all():#pixels para vermelho
            escalaRGB[i][j][0] = 229
            escalaRGB[i][j][1] = 0
            escalaRGB[i][j][2] = 28
        elif (escalaRGB[i,j]==[63,63,63]).all():#pixels para azul
            escalaRGB[i][j][0] = 48
            escalaRGB[i][j][1] = 46
            escalaRGB[i][j][2] = 192
    return escalaRGB


def eliminarPontosPretos(img):#função para eliminar os pontos pretos da imagem
    axs[1].set_title('Imagem com pontos pretos retirados')
    axs[1].set_axis_off()#tira o eixo x e y das imagem que fica na coluna 1
    fechamento = morphology.area_closing(img,64,1)#faz o fechamento(kernel/vizinhança=[4,4]) apenas com fig de tamanho <= 64(número de pixels) com conectividade 1
    axs[1].imshow(greyToRGB(fechamento),cmap='gray')

def preenchimentoDeBuracos(img):
    axs[1].set_title('Imagem com buracos preenchidos')
    axs[1].set_axis_off()#tira o eixo x e y das imagem que fica na coluna 1
    abertura = morphology.area_opening(img,800,1)#faz a abertura(kernel=[4,4]) apenas com buracos de tamanho <= 800
    axs[1].imshow(greyToRGB(abertura),cmap='gray')

def fechoConvexo(img):
    axs[1].set_title('Imagem com fecho convexo dos objetos')
    axs[1].set_axis_off()#tira o eixo x e y das imagem que fica na coluna 1
    fechamento = morphology.area_closing(img,64,1)#retira os pontos para fazer o fecho apenas das figuras restantes
    linha,coluna = fechamento.shape
    for i in range(linha):
        for j in range(coluna):               
            if (fechamento[i][j] == 88):#seta os pixels da fig de cor magenta(na escala de cinza) para branco
                fechamento[i][j] = 255
    thresh = threshold_otsu(fechamento)#auxilia na transformação da img na escola de cinza para binária(0-ausencia de cor 1-presença de cor)
    binary = invert(fechamento > thresh)#tranforma a imagem em binaria em seguida faz a troca das cor dos pixels
    fechoConvexo = morphology.convex_hull_object(binary)#retorna o fecho convexo de cada objeto
    axs[1].imshow(fechoConvexo,cmap='gray')

def hitOrMissMagenta(img):
    axs[0][1].set_title('Imagem original binária com inversão dos pixels')
    axs[1][0].set_title('Elemento Estruturante de cor magenta')
    axs[1][1].set_title('Localização do Hit or Miss para objeto de cor magenta')
    axs[0][1].set_axis_off()#tira o eixo x e y das imagem que fica na linha 0 coluna 1
    axs[1][0].set_axis_off()#tira o eixo x e y das imagem que fica na linha 1 coluna 0
    axs[1][1].set_axis_off()#tira o eixo x e y das imagem que fica na linha 1 coluna 1
    fechamento = morphology.area_closing(img,64,1)#retira todos os pontos
    linha,coluna = fechamento.shape
    for i in range(linha):
        for j in range(coluna):               
            if (fechamento[i][j] != 88):#seta os pixels da fig de cor não magenta(na escala de cinza) para branco
                fechamento[i][j] = 255
    thresh = threshold_otsu(img)#auxilia na transformação da img na escola de cinza para binária// threshold automatico
    binary = invert(img > thresh).astype(np.int) #tranforma a imagem que tem a figura magenta em binaria(true/false)
    axs[0][1].imshow(binary,cmap='gray')#mostra a imagem original binaria(com pixels invertidos)
    thresh2 = threshold_otsu(fechamento)#auxilia na transformação da img na escala de cinza para binária
    binary2 = invert(fechamento > thresh2).astype(np.int)#tranforma a imagem que tem a figura magenta em binaria(true/false)
    binary2 = binary2[~np.all(binary2 == 0, axis=1)]#deleta as linhas que tem somente zero
    idx = np.argwhere(np.all(binary2[..., :] == 0, axis=0))#procura por tds colunas que tem somente zero
    binary2 = np.delete(binary2, idx, axis=1)#deleta tds colunas que tem somente zero
    axs[1][0].imshow(binary2,cmap='gray')#mostra o elementro estruturante na forma binária
    binary2[binary2==0] = 2 #pixels de n° zero sao setados para 2(dont care) não precisa ser comparado
    hitOrMiss = mh.hitmiss(binary, binary2)#operação hit or miss para objeto de cor magenta (retorna o ponto central do objeto localizado)
    axs[1][1].imshow(hitOrMiss,cmap='gray')

def esqueletoVermelho(img):
    axs[1].set_title('Esqueleto da imagem de cor vermelha')
    axs[1].set_axis_off()#tira o eixo x e y das imagem que fica na coluna 1
    fechamento = morphology.area_closing(img,64,1)#retira todos os pontos
    linha,coluna = fechamento.shape
    for i in range(linha):
        for j in range(coluna):               
            if (fechamento[i][j] != 72):#seta os pixels da fig de cor não vermelha(na escala de cinza) para branco
                fechamento[i][j] = 255
    thresh = threshold_otsu(fechamento)#auxilia na transformação da img na escola de cinza para binária// threshold automatico
    binary = invert(fechamento > thresh).astype(np.int) #tranforma a imagem que tem a figura magenta em binaria(true/false)
    esqueleto = morphology.skeletonize(binary)
    axs[1].imshow(esqueleto,cmap='gray')

def esqueletoFechoConvexoVermelho(img):
    axs[1].set_title('Esqueleto do fecho convexo da imagem de cor vermelha')
    axs[1].set_axis_off()#tira o eixo x e y das imagem que fica na coluna 1
    fechamento = morphology.area_closing(img,64,1)#retira todos os pontos
    linha,coluna = fechamento.shape
    for i in range(linha):
        for j in range(coluna):               
            if (fechamento[i][j] != 72):#seta os pixels da fig de cor não vermelha(na escala de cinza) para branco
                fechamento[i][j] = 255
    thresh = threshold_otsu(fechamento)#auxilia na transformação da img na escola de cinza para binária(0-ausencia de cor 1-presença de cor)
    binary = invert(fechamento > thresh)#tranforma a imagem em binaria em seguida faz a troca das cor dos pixels
    fechoConvexo = morphology.convex_hull_object(binary)#retorna o fecho convexo de cada objeto
    esqueleto = morphology.skeletonize(fechoConvexo)
    axs[1].imshow(esqueleto,cmap='gray')    

try:
    img = cv2.imread("morfologia.png")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)#converte de bgr para rgb
    escalaCinza = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)#transforma img para escala de cinza
    if(img==None):
        print('Imagem morfologia.png não está no diretório corrente!')
except ValueError:
    print("Selecione uma Opção válida!\nAperte Ctrl+c para sair\n")
    while(1):
        try:
            k = int(input("1-Eliminar todos os pontos pretos\n"
                      "2-Preencher os buracos dos objetos: vermelho, verde e magenta\n"
                      "3-Encontrar o fecho convexo dos objetos: azul,vermelho e verde\n"
                      "4-Utilizar a transformada hit-or-miss para localizar cada um dos objetos de cor magenta\n"
                      "5-Esqueleto da imagem de cor vermelha\n"
                      "6-esqueleto do fecho convexo obtido a partir da imagem de cor vermelha.\n\n-> "))
            if(k<1 or k>6):
                raise(ValueError)
            if(k==1):
                fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(13, 6))#matriz de gráficos que serão exibidos
                imgOriginal(img)
                eliminarPontosPretos(escalaCinza)
            elif(k==2):
                fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(13, 6))#matriz de gráficos que serão exibidos
                imgOriginal(img)
                preenchimentoDeBuracos(escalaCinza)
            elif(k==3):
                fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(13, 6))#matriz de gráficos que serão exibidos
                imgOriginal(img)
                fechoConvexo(escalaCinza)
            elif(k==4):
                fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(13, 6))#matriz de gráficos que serão exibidos
                imgOriginal(img)
                hitOrMissMagenta(escalaCinza)
            elif(k==5):
                fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(13, 6))#matriz de gráficos que serão exibidos
                imgOriginal(img)
                esqueletoVermelho(escalaCinza)
            else:
                fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(13, 6))#matriz de gráficos que serão exibidos
                imgOriginal(img)
                esqueletoFechoConvexoVermelho(escalaCinza)
            plt.show()#mostra todos histogramas e imagens 
        except KeyboardInterrupt:
            print('Programa finalizado...')
            break
        except ValueError:
            print('Opção Inválida!\n')
