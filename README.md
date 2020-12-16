# IA376J_final_project

## 17/12/2020

# Introdução
Inicialmente a ideia era melhorar o pré-treinamento sobre os textos sintéticos usando a Wikipedia. Após dificultar a tarefa, um novo pré-treino não convergiu após alguns dias. Decidi mudar a atenção para a tarefa do DocVQA. 

# O que já foi feito:
Implementação de dataset abstraindo o DocVQA e as informações de OCR que vem com ele. 
Implementação de treino com possibilidade de envolvimento do LayoutLM no futuro (código ainda incompleto/com bugs no transformers). 

# Planejamento: 

* Baseline do que o T5-base consegue extrair das informações de OCR, sem involver informações 2D.

* Involver informação 2D, duas opções: usar saida do LayoutLM como Input Embbedings do T5, ou modificar o T5 para usar Embeddings 2D, de forma semelhante a forma que o LayoutLM faz. 

* Finalmente, involver features de imagens. 

* Analisar se houve ganho em relação ao baseline. 

* Se possível, tentar tirar a dependência do OCR com o CNNT5 (os pesos pré-treinados em imagens sintéticas) na entrada. 

