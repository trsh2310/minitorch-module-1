# MiniTorch Module 1

<img src="https://minitorch.github.io/minitorch.svg" width="50%">

* Docs: https://minitorch.github.io/

* Overview: https://minitorch.github.io/module1/module1/


## Task 1.5: Scalar training

Для четырёх основных датасетов использованы `PTS=50`, `RATE=0.5`,
`EPOCHS=500`, `SEED=0`. Simple использует `HIDDEN=2`, остальные — `HIDDEN=10`.


###  логи обучения

#### Simple файл: [Simple log](project/training_logs/simple.txt).

```text
Dataset=Simple PTS=50 HIDDEN=2 RATE=0.5 EPOCHS=500 SEED=0
Epoch  10  loss  32.406769431080704 correct 33
Epoch  20  loss  31.75779084323516 correct 33
Epoch  30  loss  30.917808732567806 correct 33
Epoch  40  loss  29.11990739341107 correct 33
Epoch  50  loss  25.807573086055683 correct 33
Epoch  60  loss  22.61811136733391 correct 43
Epoch  70  loss  19.61735009847903 correct 47
Epoch  80  loss  19.683965661735407 correct 46
Epoch  90  loss  13.188118830115254 correct 47
Epoch  100  loss  11.388223786431306 correct 49
Epoch  110  loss  15.55208986966353 correct 43
Epoch  120  loss  11.074433504142078 correct 47
Epoch  130  loss  11.75021278454066 correct 46
Epoch  140  loss  11.24055519390787 correct 46
Epoch  150  loss  10.585628747257099 correct 46
Epoch  160  loss  10.09588252398276 correct 47
Epoch  170  loss  9.901112928012916 correct 46
Epoch  180  loss  9.445798765039982 correct 46
Epoch  190  loss  8.740894820727345 correct 47
Epoch  200  loss  8.523415818130934 correct 46
Epoch  210  loss  12.053630412307175 correct 45
Epoch  220  loss  6.176262823848839 correct 48
Epoch  230  loss  9.780996473439362 correct 46
Epoch  240  loss  8.951911989357423 correct 46
Epoch  250  loss  7.9158803889081915 correct 46
Epoch  260  loss  6.573378547048167 correct 48
Epoch  270  loss  8.560544182140001 correct 46
Epoch  280  loss  5.181999618265537 correct 48
Epoch  290  loss  4.67329031000869 correct 49
Epoch  300  loss  4.70042885448622 correct 49
Epoch  310  loss  6.1821906130702216 correct 48
Epoch  320  loss  5.771446370762421 correct 48
Epoch  330  loss  3.529345336424394 correct 49
Epoch  340  loss  4.480783118998515 correct 49
Epoch  350  loss  7.777889738140609 correct 47
Epoch  360  loss  6.270211600608357 correct 48
Epoch  370  loss  8.334344166535779 correct 46
Epoch  380  loss  9.659733653036792 correct 46
Epoch  390  loss  4.359519838440642 correct 49
Epoch  400  loss  6.876636004996833 correct 47
Epoch  410  loss  9.64985727723244 correct 46
Epoch  420  loss  2.7761959505635185 correct 49
Epoch  430  loss  2.9414139429845902 correct 49
Epoch  440  loss  2.4669353298488135 correct 49
Epoch  450  loss  2.6500281230614857 correct 49
Epoch  460  loss  5.377856554366053 correct 48
Epoch  470  loss  3.18863368928109 correct 49
Epoch  480  loss  7.808953020595618 correct 47
Epoch  490  loss  3.8279206425312347 correct 49
Epoch  500  loss  2.4597273277606875 correct 49
Final training accuracy: 49/50 (98.0%)
```

#### Diag

Отдельный файл: [Diag log](project/training_logs/diag.txt).

```text
Dataset=Diag PTS=50 HIDDEN=10 RATE=0.5 EPOCHS=500 SEED=0
Epoch  10  loss  11.506140051062092 correct 47
Epoch  20  loss  10.108439714978914 correct 47
Epoch  30  loss  8.521821739350427 correct 47
Epoch  40  loss  6.871470067911914 correct 47
Epoch  50  loss  5.383166957002022 correct 47
Epoch  60  loss  4.122179034062345 correct 47
Epoch  70  loss  3.208435650723288 correct 49
Epoch  80  loss  2.584753733322638 correct 50
Epoch  90  loss  2.1372539891944604 correct 50
Epoch  100  loss  1.8170157287438835 correct 50
Epoch  110  loss  1.574740028753218 correct 50
Epoch  120  loss  1.385262935286957 correct 50
Epoch  130  loss  1.2309087057967223 correct 50
Epoch  140  loss  1.104890541649586 correct 50
Epoch  150  loss  0.9925565884224938 correct 50
Epoch  160  loss  0.8986737137308746 correct 50
Epoch  170  loss  0.8143650186044888 correct 50
Epoch  180  loss  0.7434801542577648 correct 50
Epoch  190  loss  0.679134792848688 correct 50
Epoch  200  loss  0.624017808345306 correct 50
Epoch  210  loss  0.5752180548988566 correct 50
Epoch  220  loss  0.5309793041823196 correct 50
Epoch  230  loss  0.4918443670836965 correct 50
Epoch  240  loss  0.4574471205455807 correct 50
Epoch  250  loss  0.426487997346265 correct 50
Epoch  260  loss  0.3989133115484559 correct 50
Epoch  270  loss  0.3737225059954345 correct 50
Epoch  280  loss  0.35113092118201694 correct 50
Epoch  290  loss  0.3306628675460628 correct 50
Epoch  300  loss  0.31190944345422955 correct 50
Epoch  310  loss  0.29502109877895377 correct 50
Epoch  320  loss  0.27945537579387325 correct 50
Epoch  330  loss  0.26516755184157537 correct 50
Epoch  340  loss  0.25207188369239064 correct 50
Epoch  350  loss  0.2400100729273576 correct 50
Epoch  360  loss  0.22890867650574356 correct 50
Epoch  370  loss  0.21858066445998817 correct 50
Epoch  380  loss  0.2089740408552703 correct 50
Epoch  390  loss  0.20009204644961492 correct 50
Epoch  400  loss  0.19184558424504483 correct 50
Epoch  410  loss  0.1841700928922852 correct 50
Epoch  420  loss  0.1769724153883684 correct 50
Epoch  430  loss  0.17019426079889252 correct 50
Epoch  440  loss  0.16387928406710425 correct 50
Epoch  450  loss  0.15797435419356423 correct 50
Epoch  460  loss  0.15236095203084138 correct 50
Epoch  470  loss  0.1471105410111424 correct 50
Epoch  480  loss  0.14216413658795832 correct 50
Epoch  490  loss  0.13749275968322666 correct 50
Epoch  500  loss  0.13305397887734519 correct 50
Final training accuracy: 50/50 (100.0%)
```

#### Split

Отдельный файл: [Split log](project/training_logs/split.txt).

```text
Dataset=Split PTS=50 HIDDEN=10 RATE=0.5 EPOCHS=500 SEED=0
Epoch  10  loss  29.244108529739528 correct 44
Epoch  20  loss  24.588360849677493 correct 40
Epoch  30  loss  25.318955010096357 correct 37
Epoch  40  loss  25.397897035996394 correct 37
Epoch  50  loss  22.661593225979995 correct 39
Epoch  60  loss  22.636821159609934 correct 39
Epoch  70  loss  20.96417884541807 correct 40
Epoch  80  loss  18.740234604537076 correct 41
Epoch  90  loss  24.231875160033642 correct 36
Epoch  100  loss  14.8589399011226 correct 43
Epoch  110  loss  17.922048674739393 correct 43
Epoch  120  loss  13.161609637657168 correct 46
Epoch  130  loss  10.741802525796713 correct 47
Epoch  140  loss  9.13586693608296 correct 48
Epoch  150  loss  10.776892435036705 correct 47
Epoch  160  loss  8.406931477153481 correct 48
Epoch  170  loss  7.093365776603094 correct 49
Epoch  180  loss  9.352117071962738 correct 47
Epoch  190  loss  7.896365748048278 correct 49
Epoch  200  loss  6.174417314022056 correct 49
Epoch  210  loss  5.194967851196328 correct 49
Epoch  220  loss  4.76213038018594 correct 49
Epoch  230  loss  24.73133812491171 correct 39
Epoch  240  loss  5.294683573846967 correct 49
Epoch  250  loss  4.38263919609252 correct 49
Epoch  260  loss  6.223695677244532 correct 48
Epoch  270  loss  4.212584581404797 correct 49
Epoch  280  loss  6.988606718739054 correct 47
Epoch  290  loss  4.0799804496909164 correct 49
Epoch  300  loss  4.596866486081433 correct 49
Epoch  310  loss  4.925906938232692 correct 49
Epoch  320  loss  3.511299151996488 correct 49
Epoch  330  loss  3.743015232512371 correct 49
Epoch  340  loss  3.7842573795029706 correct 49
Epoch  350  loss  3.4895166104610507 correct 49
Epoch  360  loss  4.244001923481009 correct 49
Epoch  370  loss  11.967130489177183 correct 44
Epoch  380  loss  2.807804390924093 correct 50
Epoch  390  loss  3.247613503879785 correct 49
Epoch  400  loss  3.2988005585770765 correct 49
Epoch  410  loss  2.963902184990668 correct 49
Epoch  420  loss  3.0526532590334345 correct 49
Epoch  430  loss  2.7773665167573665 correct 49
Epoch  440  loss  2.911602532520939 correct 49
Epoch  450  loss  2.8072581025698424 correct 49
Epoch  460  loss  3.463020181252319 correct 49
Epoch  470  loss  2.5815622969129457 correct 49
Epoch  480  loss  2.613861782447994 correct 49
Epoch  490  loss  3.4263669043979226 correct 49
Epoch  500  loss  2.5018063058183 correct 49
Final training accuracy: 49/50 (98.0%)
```

#### Xor

Отдельный файл: [Xor log](project/training_logs/xor.txt).

```text
Dataset=Xor PTS=50 HIDDEN=10 RATE=0.5 EPOCHS=500 SEED=0
Epoch  10  loss  29.283800687878777 correct 41
Epoch  20  loss  25.5927410002117 correct 41
Epoch  30  loss  35.705578814356144 correct 24
Epoch  40  loss  24.726062744159286 correct 37
Epoch  50  loss  23.620062127060155 correct 38
Epoch  60  loss  21.65848740157935 correct 40
Epoch  70  loss  19.36403247647846 correct 42
Epoch  80  loss  16.929288405693566 correct 42
Epoch  90  loss  14.79038963644281 correct 46
Epoch  100  loss  12.49344304395789 correct 46
Epoch  110  loss  11.696331971438319 correct 46
Epoch  120  loss  10.705348431810773 correct 46
Epoch  130  loss  10.605136558263235 correct 46
Epoch  140  loss  11.417851616496582 correct 46
Epoch  150  loss  10.336880210589984 correct 46
Epoch  160  loss  8.736617984659144 correct 46
Epoch  170  loss  8.246646383058387 correct 46
Epoch  180  loss  8.43685866047406 correct 46
Epoch  190  loss  9.598849696700666 correct 46
Epoch  200  loss  10.178642555063568 correct 46
Epoch  210  loss  8.48490023542385 correct 46
Epoch  220  loss  7.802823657632472 correct 46
Epoch  230  loss  7.685737781236042 correct 46
Epoch  240  loss  7.825906077725532 correct 46
Epoch  250  loss  7.3218866976299815 correct 47
Epoch  260  loss  6.890887495502532 correct 47
Epoch  270  loss  6.741540482719666 correct 47
Epoch  280  loss  6.676601588437733 correct 47
Epoch  290  loss  6.6016240979550025 correct 47
Epoch  300  loss  6.426869898880135 correct 47
Epoch  310  loss  6.3171515434292145 correct 47
Epoch  320  loss  6.23801163593653 correct 47
Epoch  330  loss  6.14200064273287 correct 47
Epoch  340  loss  6.022331845921715 correct 48
Epoch  350  loss  5.980042283328696 correct 48
Epoch  360  loss  5.709275034432039 correct 48
Epoch  370  loss  5.646486953190586 correct 48
Epoch  380  loss  5.532755260898227 correct 48
Epoch  390  loss  5.542431600579354 correct 48
Epoch  400  loss  5.433295671467095 correct 48
Epoch  410  loss  5.393828153513913 correct 48
Epoch  420  loss  5.343325969621857 correct 48
Epoch  430  loss  5.375427716575345 correct 48
Epoch  440  loss  5.240661266659943 correct 48
Epoch  450  loss  5.297179213974181 correct 48
Epoch  460  loss  4.6337722576703015 correct 48
Epoch  470  loss  4.982264980486426 correct 48
Epoch  480  loss  4.975198366394294 correct 48
Epoch  490  loss  4.87430194064597 correct 48
Epoch  500  loss  4.65646724102726 correct 48
Final training accuracy: 48/50 (96.0%)
```
