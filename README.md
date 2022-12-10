# Bayesian_methods_Context_Trees

# VLMC simulation

This code is written in Python and simulates a stochastic chain with memory of variable length accepting any number of contexts and states.

In order for the code to work properly, we must respect the following conventions: 

- The context matrix ("cont_mat"), which is a parameter of most functions throughout the code must have its contexts written from right to left and if the number of elements in a given context is less than the greatest size of a context, the remaining entries of that row are filled with None objects.

- The probabilities of each row of the transition matrix correspond to the context of same row number of the context matrix.

Now, a brief description of the functions present in the code:

- extract_context: Given an array which begins with null elements (represented by the keyword "None" in python), the function returns an array with only the elements which are not null of the array passed as a parameter.

- context_row: Relates the context of the chain to the respective row of the context matrix so the next function can compute the next state of the chain.
The parameters are the array with the last n elements of the chain, where n is the greatest length of a context and the context matrix.

- next_state: Computes the next state of the chain, but doesn't return anything. Its parameters are: transition matrix, current context, context matrix, whole chain, current index of the chain array and the array of the uniforms (for checking the results).

- simulate_sample: The main function of the code. Its parameters are the length of the chain, the transition matrix, the initial context to kick-start the chain and the context matrix. To call the function write simulate_sample(n, P, mat_cont).

# Bayesian Methods

What follows is a detailed explanation of all the functions and classes present in the code (in portuguese):

Nos códigos que vamos descrever, ao invés de calcularmos alguma probabilidade, calcularemos o logaritmo dessa probabilidade. Este artíficio foi usado também pelos autores do artigo no qual nos baseamos (Kontoyiannis et al. (2022)). Ele se faz necessário, pois conforme multiplicamos as probabilidades, os números resultantes se tornam cada vez menores, ao ponto de a precisão do computador não ser suficiente para obtermos um valor que seja diferente de zero. Com o logaritmo das probabilidades conseguimos contornar este problema, somando esses valores, sempre que houver uma multiplicação das probabilidades. Para o caso em que há soma de probabilidades, como é o caso do cálculo da probabilidade ponderada, $P_{w,\lambda}$, usamos um algoritmo para calcular exatamente o logaritmo da soma, que foi encontrado na internet e que será descrito na subseção da classe Node.

## Implementação do algoritmo CTW

Para a implementação do algoritmo CTW usamos programação orientada a objetos, criando duas classes. Uma classe chamada CTW e outra chamada Node. Para cada amostra cuja árvore de contextos desejamos estimar, é criada uma única instância da classe CTW e uma instância da classe Node para cada nó presente na árvore maximal.

### Classe CTW

Os atributos da classe CTW são:


- Os parâmetros do algoritmo de mesmo nome;
    
- O dicionário que guarda os nós da árvore maximal. É definido como um dicionário vazio;
    
- A matriz das folhas que estão na profundidade maximal, ou seja, de todos os contextos diferentes de tamanho igual à maior profundidade da árvore maximal. É calculada usando um laço $\textit{while}$, obtendo uma fatia de tamanho $D$ da sequência e movendo em uma unidade a localização dessa fatia a cada iteração;
    
- O endereço da instância do nó raíz. É obtido e adicionado ao dicionário como todos os outros nós: é criado pela classe Node e adicionado com o método add_node, do qual falaremos adiante;
    
- Uma lista das strings que representam todas as folhas da árvore maximal, e que são iguais às chaves do dicionário que guarda os nós da árvore maximal. Para obter este, simplesmente percorremos todas as chaves do dicionário com um laço $\textit{for}$.


O parâmetro $\beta$ tem como valor padrão $\textbf{None}$ (o objeto nulo do Python), neste caso beta é igual a $1-2^{(-m + 1)}$, onde m é o tamanho do alfabeto.

Os métodos são:


- add_node: Adiciona o nó cujo contexto recebe como parâmetro à árvore (ao dicionário que guarda os elementos da árvore);
    
- aux_tree: Auxilia nos cálculos dos nós que entrarão na árvore maximal usando o método add_node;
    
- seq_update: Não recebe nenhum parâmetro. Realiza a atualização sequencial percorrendo a cadeia a partir do $(D + 1)$-ésimo elemento da amostra.
    


#### Funcionamento da classe

Em uma chamada da classe CTW, além da definição dos atributos, há o cálculo dos nós da árvore maximal usando o método $\textit{aux\_tree}$, com $lD$ iterações, onde $l$ é o número de folhas que há na profundidade maximal. Após isso, chamamos o método $\textit{get\_children\_parent}$ da classe Node, a partir do nó raíz, que constrói a relação entre pais e filhos dos nós da árvore. Estas relações são necessárias para os cálculos da probabilidade ponderada na raíz, $P_{w,\lambda}$ e para a probabilidade maximal, $P_{m,\lambda}$. Em seguida, obtemos a lista de folhas da árvore. A seguir, calculamos os vetores de contagem e os logaritmos das probabilidades estimadas dos contextos para as $D$ primeiras observações da amostra. Então, efetuamos esses mesmos cálculos para o resto da sequência, mas utilizando a atulização sequencial pelo método $\textit{seq\_update}$, como descrito na seção (\ref{sec 3.4}). Por último, usando o método $\textit{weighted\_prob}$ da classe Node, calculamos a probabilidade ponderada na raíz, $P_{w,\lambda}$, recursivamente, começando pelo nó raíz.

### Classe Node 

Os atributos da classe Node são:


- Os seus parâmetros: a instância da classe ctw e o array do contexto;
    
- O vetor de contagens iniciado com zeros;
    
- O nó pai do contexto passado como parâmetro, que será um objeto e que é iniciado com $\textbf{None}$;
    
- A soma dos elementos do vetor de contagens, $M_s$, inicializado com zero;
    
- O logaritmo da probabilidade estimada;
    
- O logaritmo da probabilidade ponderada;
    
- O logaritmo da probabilidade maximal para o algoritmo BCT;
    
- Uma variável que indica se o primeiro termo é o máximo da expressão $(\ref{3.2})$ no passo $(ii)$ do algoritmo BCT. Este atriibuto é para ser usado na implementação do passo $(iii)$, no método $\textit{prune\_tree}$ da classe BCT;
    
- Uma variável booleana, que nos diz se o nó é o raíz;
    
- E um dicionário vazio, para guardar os filhos do nó.
    


E seus métodos são:



- get_children_parent: obtém os pais e filhos de todos os nós, podendo ser chamado de qualquer nó (na classe CTW fazemos isso do nó raíz);

- get_node_children: Obtém os filhos somente do nó a partir do qual está sendo chamado. Este método é útil na implementação do amostrador MCMC;

- estimated_prob: Calcula o logaritmo da probabilidade estimada do nó, como descrito no passo $(iii)$ da subseção (\ref{CTW sec}), para apenas os $D$ primeiros elementos da amostra;
    
- weighted_prob: recebe como parâmetro a constante $\beta$ e calcula a probabilidade ponderada do nó de forma recursiva;
    
- log_sum: calcula exatamente o logaritmo natural da soma em função dos logaritmos dos somandos com base na seguinte igualdade: $ln(a + b) = ln\{exp[ln(a) - ln(b)] + 1\} + ln(b)$;
    
- next_update: recebe o parâmetro $\beta$ e o índice do elemento que sucede o contexto do nó na sequência. Calcula a atualização do logaritmo da probabilidade estimada do nó. É usado no método $\textit{seq\_update}$ da classe CTW;
    
- maximal_probability: calcula a probabilidade maximal do nó, como descrito no algoritmo BCT. É usado na classe BCT. 
    


A classe Node, além de auxiliar a CTW na execução do algoritmo tem atributos e métodos referentes à classe BCT, que é a implementação do algoritmo homônimo e a qual descreveremos a seguir.

## Implementação do algoritmo BCT

Essa implementação também é feita com a ajuda da classe Node, como foi descrito, pois uma vez usada a programação orientada a objetos, é muito difícil dissociar os nós de suas propriedades descritas nos algoritmos CTW e BCT. Além disto, o emprego desta técnica de programação é intuitiva na medida em que se torna necessário obtermos as relações de pais e filhos entre os nós ao longo das implementações.\\

A classe BCT recebe como parâmetros os mesmos do algoritmo CTW. Seus atributos são:


\begin{itemize}
    \item A instância do CTW referente à amostra;
    
    \item As strings que representam as novas folhas da árvore podada;
    
    \item A probabilidade a posteriori da árvore estimada.
    
\end{itemize}

Seus métodos são:

\begin{itemize}
    \item $\textit{prune\_tree}$: poda as folhas até chegar na árvore mais provável, de forma recursiva. Ele recebe o nó a partir do qual começamos a podar (na primeira chamada será sempre o nó raíz) e a profundidade da árvore a partir da qual podemos podar.;
    
    \item $\textit{tree\_is\_equal}$: compara a árvore estimada pelo método com a árvore recebida como parâmetro. Essa árvore é uma lista de strings que representam as folhas. Esse método é usado em algumas análises presentes no relatório. 
    
\end{itemize}


\section{Implementação do algoritmo MCMC}

Para as diferentes árvores que são criadas quando removemos e colocamos os filhos de um nó, criamos a classe $\textit{new\_Tree}$ para instanciar essas árvores mais duas outras funções.

\subsection{Classe \texorpdfstring{$\textit{new\_Tree}$}{} }

Esta classe recebe uma instância da classe CTW (pois iremos precisar da árvore maximal e de outros de seus atributos) e o dicionário contendo os nós da nova árvore. Seus atributos são:

\begin{itemize}
    \item O dicionário da árvore;
    
    \item A instância de CTW
    
    \item O número de nós da árvore;
    
    \item Uma lista das strings que representam os contextos das folhas;
    
    \item O número de folhas da árvore na profundidade $D$;
    
    \item Uma lista das strings dos contextos com apenas $m$ descendentes, onde $m$ é o tamanho do alfabeto. 
    
\end{itemize}

Seus métodos são:

\begin{itemize}
    \item $\textit{add\_children}$: devolve uma nova instância de $\textit{new\_Tree}$ com os filhos do nó passado como parâmetro adicionados;
    
    \item $\textit{remove\_children}$: devolve uma nova instância de $\textit{new\_Tree}$ com os filhos do nó passado como parâmetro removidos;
    
    \item $\textit{compares\_trees}$: compara a árvore passada como parâmetro e a árvore $\textit{self}$, aquela a partir da qual o método é chamado. Retorna $\textbf{True}$ se forem iguais e $\textbf{False}$, caso contrário;
    
    \item $\textit{leaves}$: retorna a lista das strings que representam os contextos das folhas;
    
    \item $\textit{leaves\_D}$: retorna o número de folhas da árvore na profundidade $D$;
    
    \item $\textit{internal\_D}$: retorna a lista das strings dos contextos com apenas $m$ descendentes;
    
    \item $\textit{num\_bayes}$: retorna o logaritmo do numerador da fórmula de bayes como explicitado na última razão da equação $(\ref{3.5})$.
    
\end{itemize}

\subsection{Outras funções do algoritmo}

\begin{itemize}
    \item $\textit{mult\_ratio}$: ela auxilia no cálculo da probabilidade de aceitação, calculando o número a ser multiplicado pela razão das probabilidades, como descrito pela equação $(\ref{eq 3.6})$. Recebe a instância da árvore que é o elemento atual da sequência, o novo elemento proposto para a sequência, a instância da classe CTW, o número máximo de nós que uma árvore com a profundidade maximal igual a da árvore maximal pode ter e uma variável de controle para nos dizer de que forma a nova árvore proposta foi formada, indicada no passo $(i)$ do algoritmo;
    
    \item $\textit{RW\_sampler}$: Nesta função usamos os atributos e métodos da classe $\textit{new\_Tree}$ para gerar a amostra, que é devolvida por ela.  Ela recebe a árvore inicial, que é formada por elementos da árvore maximal, o tamanho da amostra a ser gerada e a instância do CTW, ou os próprios parâmetros da classe CTW, que pode ser instanciada dentro da própria função. No caso de ser usada a mesma instância do CTW como parâmetro em várias chamadas, é preciso se atentar para o fato de a nossa implementação modificar a árvore maximal e, assim, implementar também um $\textit{reset}$ nela, definindo os filhos das folhas como o dicionário vazio.
    
\end{itemize}

Uma outra função, que não é extamente do algoritmo, mas que é essencial para estimarmos as probabilidades a posteriori das árvores da cadeia é $\textit{time\_list}$. Ela recebe a amostra gerada por $\textit{RW\_sampler}$ e retorna uma lista em que cada elemento é uma lista de três outros elementos: a proporção de tempo em que a cadeia passa em uma árvore, uma lista de strings dos contextos das folhas dessa árvore e o objeto da árvore correspondente. A lista maior é ordenada de forma decrescente com relação às proporções.

A última função desse bloco é a $\textit{Bayesian\_MCMC}$. Ela recebe os mesmos argumentos que $\textit{RW\_sampler}$ e retorna a mesma lista que a função $\textit{time\_list}$ retorna.

