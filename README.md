# Introduction to text classification

## La nascita del NLP

I big data rendono possibile l’apprendimento. Al giorno d’oggi viviamo in un mondo costantemente connesso e ogni giorno vengono elaborati e memorizzati grandi quantità di dati. Grazie ai big data è nato il NLP, ovvero l’applicazione di tecniche di Machine Learning per cercare di capire e trovare una rappresentazione per testo scritto o parlato. Infatti per applicare questo tipo di modelli e riuscire ad apprendere dei pattern nei documenti è importante essere in possesso di grandi quantità di dati.

Tra i principali task di classificazione di testi ritroviamo la classificazione di testi e la sentiment analysis:

- Classificazione di testi: consiste nell’apprendere la categoria di un documento a partire dal testo del documento;
- Sentiment analysis: consiste nell’apprendere l’opinione di un utente, ad esempio su un prodotto, sulla base di quello che ha scritto sul prodotto;
- Rate prediction: consiste nell’apprendere il punteggio che l’utente da ad un oggetto sulla base di ciò che ha scritto nella recensione.

## Text classification task

- Dati: dataset di documenti suddivisi in varie categorie, ad esempio auto, informatica e calcio;

- Obiettivo: dato un nuovo documento mai osservato attribuirgli una categoria tra le tre prefissate;

- Apprendimento: durante il training vengono esaminati tutti i documenti del dataset. L’algoritmo apprendere le seguenti tipologie di pattern:

- - Categoria auto: le parole discriminanti sono benzina, diesel, cofano, cilindrata, ecc;
  - Categoria calcio: le parole discriminanti sono calciatore, stadio, partita, ecc;
  - Categoria informatica: le parole discriminanti sono pc, computer, programming, java, hw, sw, ecc.

​	Dopo aver finito la fase di apprendimento e aver valutato in maniera intelligente e pensata le performance del classificatore è possibile iniziare a predirre categorie per nuovi documenti mai osservati. Il classificatore semplicemente prende in input il nuovo documento e restituisce la sua classe. Per far questo si basa sulle parole discriminanti che incontra nel corpo del documento. Ad esempio se trova molte parole discriminanti per la categoria auto restituirà la classe auto.

# FastText Architecture

## Word embeddings

Poiché un algoritmo riesca a lavorare con delle parole, queste devono essere trasformate in una rappresentazione che la macchina riesce a comprendere. Quindi prima di essere fornite in input ad un qualsiasi algoritmo le parole vengono trasformate in vettori. Questi vettori, nel caso di FastText e in molti casi di NLP, prendono il nome di word embeddings. Il word embedding non è un semplice vettore one-hot encoded (vettori sostituiti dai word embeddings) ma un vettore costruito in modo ragionato, in modo che vettori che rappresentano parole simili siano vicini nello spazio. L’idea è che più questi vettori sono vicini nello spazio e più le parole che essi rappresentano sono simili. Ma cosa si intende vicini nello spazio? Per dare un interpretazione dobbiamo pensare al vettore come coordinate in uno spazio 2D e rappresentare il vettore su un piano cartesiano. Vogliamo costruire questi vettori in modo che parole simili siano rappresentate da vettori che nel piano sono molto vicini tra loro, mentre parole dissimili siano rappresentate da vettori che nel piano sono molto distaccati tra loro. Per fare un esempio la parola Ferrari è di classe diversa rispetto alla parola hamburger, quindi i rispettivi vettori saranno distanti sul piano, mentre la parola hamburger è molto simile alla parola cheeseburger e quindi i vettori di queste due parole saranno molto vicini sul piano. Solitamente nel NLP si parla di apprendimento di word embeddings, infatti è impossibile costruire dei vettori con queste proprietà da zero. Modelli che permettono l’apprendimento di word embeddings sono i word2vec models: skip-gram e continuos bag of words, da cui deriva il classificatore di FastText.

## Apprendimento di word embeddings

I word embeddings sono solitamente appresi da un modello word2vec, ossia un modello che prende in input un insieme di parole e produce in output un vector space, dove ogni singola parola del documento di input viene fatta corrispondere ad un vettore nello spazio, detto appunto word embedding. I word embeddings prodotti cattureranno le relazioni tra le parole passate in input al modello. 

I vettori saranno predisposti nello spazio in maniera tale che le parole che condividono contesti comuni nel documento di input saranno rappresentate da vettori vicini nello spazio.

Due implementazioni di word2vec che permettono di apprendere i word embeddings sono:

1. Skip-gram model: data una parola in input, predice in output il contesto della parola;
2. CBOW (Continuous bag of words) model: dato un contesto di parole in input, predice in output la parola che ricade in quel contesto. L’architettura di FastText deriva dal modello CBOW.

## FastText classifier

FastText può essere usato per apprendere word embeddings oppure per fare text classification. Da ora in poi parleremo di FastText come text classifier, e quindi un algoritmo che dato un nuovo documento predice una classe per il documento. Come funziona l’apprendimento in FastText?

Bisogna pensare ai documenti e alle rispettive classi come vettori in un vector space. Le coordinate di questi vettori è quello che FastText cerca di apprendere. Le coordinate vengono apprese in modo che il vettore che rappresenta un documento sia molto vicino al vettore che rappresenta la sua label. 

Al termine dell’apprendimento le relazioni catturate dai word embeddings appresi saranno relazioni del tipo: parola - classe. L’idea è che se durante l’apprendimento una parola è capitata molte volte all’interno di documenti di classe X, allora in fase di predizione se in un documento appare tale parola sarà molto probabile che il documento venga classificato di classe X, questo perché il vettore della parola sarà molto vicino al vettore della classe X nel vector space costruito in fase di apprendimento.

## Architettura FastText

Il FastText classifier è una rete neurale con un solo hidden layer. Richiede in input le parole che compongono il documento di cui si vuole predirre la classe e fornisce in output la classe predetta per tale documento. Nelle slide successive andremo nel dettaglio di come funziona questa architettura e di quali formalismi statistici e matematici vengono usati al suo interno.

Quando viene lanciato il comando per allenare un modello FastText si richiede di fornire in input un dataset di documenti. Durante l’apprendimento viene visionato un documento alla volta. La prima cosa che viene fatta è la costruzione della bag of words del documento. In pratica si scorre sequenzialmente il documento e si costruisce una bag di tutte le parole incontrate nel documento. Questa bag è ciò che viene fornito in input alla rete. 

Tra lo strato di input e lo strato hidden vi è una matrice di lookup che contiene i word embeddings per ogni parola all’interno dell’intero vocabolario del training set. Il vocabolario viene costruito prima di iniziare l’apprendimento e in pratica è una bag of words dell’intero training set. La dimensione di questa matrice è quindi nxdim, dove è n è il numero di parole all’interno del vocabolario e dim è la dimensione dei word embeddings. I word embeddings possono essere appresi dalla rete oppure possono essere forniti dei word embeddings pre-trained. Se i word embeddings devono essere appresi dalla rete la matrice viene inizializzata con valori random, mentre se vengono forniti dei pre-trained word embeddings la matrice viene inizializzata con i pre-trained word embeddings.

La matrice di lookup serve per far corrispondere ogni parola in input al suo word embedding. Dopo aver calcolato questa corrispondenza, i word embeddings delle parole nel documento di input vengono mediati andando a costituire un nuovo word embedding che rappresenta l’intero documento. Questo vettore viene restituito in output dall’hidden layer.

Abbiamo detto che in output all’hidden layer viene fornito un vettore che rappresenta l’intero documento. Lo step successivo è calcolare gli score per ognuna delle etichette possibili del dataset. Questo viene fatto tramite una matrice situata tra hidden e output layer. Questa matrice ha dimensione dimxm, dove m è il numero di etichette all’interno del dataset. Viene effettuato il prodotto scalare tra il vettore dell’hidden layer e la matrice e vengono forniti in output gli score per ognuna delle etichette nel dataset. 

Gli score vengono poi convertiti in probabilità tramite l’applicazione della funzione softmax. In realtà la softmax non calcola delle probabilità, semplicemente normalizza i valori in input in modo che siano tutti compresi tra 0 e 1. Valori più alti di input avranno valori più alti di output. Poiché la softmax produce un insieme di valori nel range [0,1] la cui somma è 1, il suo output può essere interpretato come la probabilità di un’etichetta per il documento. Dopo l’applicazione della softmax è possibile calcolare la funzione errore. La funzione errore utilizzata in FastText è la negative Log-Likelihood (NLL), spesso utilizzata in coppia con la softmax function. 

La funzione viene utilizzata perché è facilmente interpretabile dato l’output della softmax e perché è convessa e questo garantisce la presenza di un minimo.

Durante il training del modello l’obiettivo è quello di minimizzare tale loss function, infatti più alto è il valore della loss function e più il nostro modello non riesce ad apprendere (non fornisce le etichette corrette per i documenti), più il suo valore si avvicina al minimo è più il nostro modello classifica correttamente.

Se in seguito all’applicazione della softmax, la classe corretta è caratterizzata da un’alta probabilità allora la loss function sarà bassa, mentre se la classe corretta è caratterizzata da una bassa probabilità allora la loss function sarà alta. Nei casi in cui la loss function è alta bisogna cercare di aggiustare i pesi delle matrici all’interno del modello per fare in modo che il documento venga assegnato alla classe corretta. Per far questo viene calcolato il gradiente della loss function. Il gradiente indica la direzione di massima crescita della funzione. Dopo aver calcolato il gradiente sarà sufficiente aggiornare i pesi della rete per fare in modo di andare nella direzione opposta al gradiente. Di quanto aggiornare i pesi e quindi di quanto spostarsi nella loss function è dettato dal learning rate, ovvero un iperparametro dell’algoritmo. Più il learning rate è alto e più gli aggiornamenti dei pesi saranno bruschi. Arrivati a questo punto ha inizio la fase backward dell’apprendimento. 

Durante la fase backward i pesi della matrice tra output e hidden layer vengono aggiustati per fare in modo che il vettore che rappresenta il documento si avvicini al vettore che rappresenta la classe corretta, in modo da ridurre la loss function. In questo step si cerca di apprendere la classe corretta per il documento.

Per quanto riguarda i pesi della matrice tra hidden e input layer, essi vengono aggiustati in modo da apprendere i word embeddings per le parole nel documento fornito in input. In questo modo viene catturata la relazione tra parola e classe corretta. Per esempio al termine del documento, il vettore che rappresenta la parola cilindrata sarà molto vicino al vettore che rappresenta la classe auto.

## Caratteristiche algoritmo di apprendimento

FastText utilizza l’algoritmo SGD in fase di apprendimento. Questo presenta vantaggi e svantaggi rispetto al full gradient descent. Nel FGD il gradiente viene calcolato rispetto a tutti gli esempi del training set. Questo significa che deve essere calcolato l’output della rete per tutti i documenti e poi può essere calcolato il gradiente della loss function. Il gradiente calcolato è esatto ma viene effettuato un solo aggiornamento dei pesi per epoca (per epoca si intende il passaggio di tutti gli esempi di training).

Nel SGD invece il gradiente della loss function viene calcolato rispetto ad un esempio per volta. Questo significa che se il dataset contiene 10000 esempi, vengono effettuati 10000 aggiornamenti dei pesi per epoca. Sebbene il gradiente rispetto ad un solo esempio risulti meno accurato del FGD, lo SGD rende l’apprendimento molto più veloce ed efficiente. In aggiunta si è dimostrato che se il learning rate viene scelto sufficientemente piccolo, la soluzione dello SGD converge alla soluzione del FGD.

FastText utilizza inoltre un decaying learning rate lineare. Si è dimostrato che utilizzare un learning prefissato durante l’apprendimento risulta inefficace. Questo perché se esso viene scelto troppo piccolo i tempi di apprendimento risulteranno molto lunghi perché gli spostamenti verso il minimo della loss function sono molto piccoli. Mentre se viene scelto troppo grande potrebbe essere che gli spostamenti risultino troppo bruschi e che la funzione obiettivo non venga ridotta ad ogni passo. 

Studi recenti propongono l’utilizzo di un learning rate che vari durante l’apprendimento. Una delle tecniche utilizzate è la discesa lineare del learning rate dopo ogni epoca. A questo proposito si inizializza il learning rate e l’algoritmo parte dal valore fornito ed inizia a ridurlo di epoca in epoca. Questo permette di avere aggiornamenti significativi nelle fasi iniziali dell’apprendimento per concludere con degli aggiornamenti più lievi. Infatti questo permette di evitare il problema di effettuare aggiornamenti troppo bruschi in prossimità del minimo della loss function e di saltare in un punto distante dal minimo.

## FastText vs Deep learning

Uno dei vantaggi di FastText è la velocità con cui esso apprende. Risulta più veloce rispetto ai modelli che utilizzano reti neurali deep. Questo perché il numero di parametri che devono essere aggiustati durante ogni fase backward è di gran lunga inferiore e anche il numero di operazioni e calcoli matriciali necessari per classificare un esempio.

Il secondo vantaggio è che nonostante sia più veloce, l’accuratezza ottenuta con FastText risulta in linea rispetto alle accuratezze restituite dai modelli deep. Questo perché FastText sfrutta le potenzialità dei **word embeddings** che riescono a catturare le relazioni tra classi e documenti in maniera discriminante (poca incertezza sul risultato finale, la classe viene assegnata con elevata confidenza). Questo perché questi vettori catturano relazioni sintattiche e semantiche tra le parole.

L’elevata accuratezza su alcuni dataset è anche data dal fatto che FastText può utilizzare i word n-grams in input. Apprendere dei word embeddings per i **word n-grams** permette di tenere in considerazione l’ordine in cui le parole appaiono nel documento.

# Iperparametri di FastText

## Numero di epoche

In machine learning, e più precisamente nel contesto delle reti neurali, per epoca si intende un ciclo di training che passa tutti gli esempi di training. Il numero di epoche è un iperparametro nella maggior parte dei modelli che implementano reti neurali e deve essere scelto con attenzione. Per esempio, se il mio dataset ha sufficienti esempi di training, il numero ottimale di epoche sarà relativamente basso perché il numero di aggiornamenti dei pesi della rete è sufficiente per apprendere. Mentre se il mio dataset conta un numero relativamente basso di esempi, il numero ottimale di epoche sarà più alto in quanto sarà necessario ciclare più volte su quei pochi esempi per ottenere degli aggiornamenti dei pesi della rete che portino a delle buone performance.

Questo parametro deve essere scelto con attenzione in quanto se il numero di epoche è troppo basso allora avrò delle cattive performance in quanto il modello farà underfitting, ovvero non riuscirà a catturare tutta la complessità dei dati, mentre se il numero di epoche è troppo alto potrei non apprendere più per un numero elevato di epoche, in quanto ho già raggiunto il minimo, e quindi avrei dei tempi di apprendimento molto lunghi inutili.

## Learning rate e n-grams

Come detto precedentemente il learning rate determina quanto gli aggiornamenti ai pesi della rete sono bruschi o meno e quindi controlla la velocità degli aggiornamenti dell’algoritmo. Abbiamo già parlato delle problematiche legate alla scelta del learning rate.

Normalmente questo parametro deve essere scelto tramite una procedura di cross-validation, che verrà spiegata successivamente. In pratica questa procedura allena k modelli, ognuno con un valore di learning rate diverso, e testa le performance di questi k modelli su un validation set. Alla fine viene utilizzato il modello che ha restituito le migliori performance.

L’iperparametro wordNgrams determina il numero di n-grams di cui tener conto e determina quindi l’input della rete di FastText. Di default vengono utilizzati gli uni-grams, ovvero vengono fornite alla rete tutte le parole che compongono il documento di input. Se invece si setta il parametro a 2, allora la rete prenderà in input coppie di parole, ovvero 2-grams, come si può anche vedere nell’immagine.

Questo parametro influenza particolarmente le performance del modello su alcuni dataset perché permette di apprendere anche l’ordine con cui le parole vengono scritte. In alcuni dataset l’ordine può influire sulla classe di appartenenza mentre in altri no.

Per questo motivo occorre testare le performance del modello con valori differenti di questo parametro e utilizzare il numero di n-grams che porta alle migliori performance.

## Dimensione word embeddings

La dimensione dei word embeddings è un parametro che incide molto sulle performance di un modello. La scelta della dimensione è indipendente dalla dimensione del vocabolario. Bisogna tener conto del tradeoff tra la dimensione e la facilità di utilizzo da parte del modello:

- Alta dimensionalità: catturano più regole semantiche e sintattiche tra le parole ma risultano **inefficienti** in fase di apprendimento;
- Bassa dimensionalità: catturano una rappresentazione più compatta per le parole diventando più efficienti da utilizzare in fase di appredimento.

Solitamente si scelgono valori che sono potenze di 32.

# Tuning iperparametri

Per tuning degli iperparametri di un modello si intende la procedura tramite la quale vengono identificati i valori ottimali per gli iperparametri. Per valori ottimali si intendono i valori degli iperparametri che garantiscono le migliori performance del modello. La misurazione delle performance può essere fatta tramite hold-out validation oppure con K-fold cross validation. 

Per testare tutte le combinazioni possibili degli iperparametri si utilizza la procedura grid search:

1. Si specificano i valori che devono essere testati per ogni parametro;
2. Si costruiscono tutte le combinazioni dei valori dei parametri;
3. Si allena un modello per ogni combinazione di valori sul training set;
4. Si valutano le performance del modello allenato tramite hold-out validation o k-fold cross validation;
5. Si tiene traccia dei risultati per scegliere il miglior modello al termine della procedura.

Bisogna scegliere con attenzione il numero di valori degli iperparametri da testare, poiché viene allenato un modello per ogni combinazione di questi. Se gli iperparametri sono 3 e si vogliono testare 2 valori per ogni iperparametro, il numero di modelli che devono essere allenati è 2^3=8. Supponiamo che ogni allenamento richieda un tempo fissato di 3 minuti e che venga utilizzata la procedura hold-out, allora il tempo richiesto dalla procedura grid search è 8*3=24 minuti.

## Hold-out

Nella procedura hold-out l’intero training set viene suddiviso in due insiemi di dati disgiunti:

- Una parte dei dati (solitamente il 70%) viene utilizzata per il training del modello;
- La parte restante (solitamente il 30%) viene utilizzata per valutare le performance del modello, e prende il nome di validation set.

La procedura consiste nell’allenare più modelli con differenti combinazioni di valori degli iperparametri e di testare le performance di questi modelli sul validation set. Al termine della procedura si seleziona il modello che ha riportato le migliori performance sul validation set e si allena il modello sull’intero training set.

Questa procedura ha un importante svantaggio:

1. Poiché gli esempi che compongono i due insiemi vengono pescati in maniera casuale vi è variabilità nella procedura e nei suoi risultati. La variabilità dipende appunto da quali esempi ricadono nel training set e quali nel validation set. Potrebbe essere quindi che il modello scelto ha delle buone performance su quello specifico validation set e che quando si esce da tale insieme e quindi si testa il modello su dati nuovi, esso avrà delle basse performance.

Come si può vedere nell’immagine di destra, i valori ottimali degli iperparametri dipendono dalla variabilità legata alla scelta dei due insiemi, e questo non va bene.

## K-fold cross validation

Questa procedura presenta degli importanti vantaggi rispetto alla precedente:

1. Ciascun esempio del dataset viene utilizzato sia per il training che per il testing del modello;
2. La variabilità legata alla scelta degli esempi da inserire nei k-fold è minore rispetto alla variabilità nella hold-out validation: questo permette di ottenere misurazioni di performance più affidabili. Questo può essere osservato nella seconda immagine.

È importante fare attenzione a quale metodo usare, infatti se si confrontano le procedure in termini di oneri computazionali la hold-out validation è la migliore, infatti viene allenato un modello per ogni combinazione di iperparametri. Mentre nella procedura di cross validation vengono allenati **K modelli per ogni combinazione di iperparametri**.

NB Più i fold sono piccoli e meno variabilità c’è nel metodo di validazione.