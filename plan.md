## Plano do Projeto (núcleo de padrões)

### Escopo
Somente detecção/validação de padrões e rotulagem em `src/patterns/**` e seus dados em `data/**`.
Ignorar agente/web, tools e main.

### Objetivo
Gerar datasets de padrões (foco em OCO/OCOI), rotular via GUI, revisar erros e consolidar um dataset limpo para modelos.

### Estrutura relevante
- `src/patterns/OCOs/necklineconfirmada.py`: lógica central de H&S (OCO/OCOI) e configuração de saída.
- `src/patterns/analise/anotador_gui.py`: GUI de rotulagem com ZigZag por estratégia, MACD/RSI e zoom por densidade.
- `src/patterns/analise/anotador_gui_correto.py`: visualizador robusto de predições (padroniza datas, plot com contexto).
- `src/patterns/analise/anotador_gui_erros.py`: revisão/correção de FP/FN e notas, salvando no CSV mestre.
- Dados: `data/datasets/patterns_by_strategy/` (entrada/saída do fluxo de rotulagem).

### Pré-requisitos
- Python 3.10+
- Instalar dependências: `pip install -r requirements.txt`

### Execução
- Gerar dataset (CLI):
  - Básico: `python src/patterns/OCOs/necklineconfirmada.py`
  - Filtros: `python src/patterns/OCOs/necklineconfirmada.py --tickers BTC-USD,ETH-USD --strategies swing_short,intraday_momentum --intervals 15m,1h --period 2y --output data/datasets/patterns_by_strategy/dataset_patterns_final.csv`
- Rotulagem (GUI): `python src/patterns/analise/anotador_gui.py`
  - Entrada: `data/datasets/patterns_by_strategy/dataset_patterns_final.csv`
  - Saída: `data/datasets/patterns_by_strategy/dataset_patterns_labeled.csv`
- Visualizar predições (GUI): `python src/patterns/analise/anotador_gui_correto.py`
- Revisão de erros (GUI): `python src/patterns/analise/anotador_gui_erros.py`

### Fluxo de dados
1) (Geração) Dataset de padrões OCO/OCOI a partir de séries de preço.
   - Arquivo alvo: `data/datasets/patterns_by_strategy/dataset_patterns_final.csv`
   - Em `necklineconfirmada.py`, o diretório base é `OUTPUT_DIR = 'data/datasets/patterns_by_strategy'`.
2) (Rotulagem) Abrir `anotador_gui.py` → produzir `dataset_patterns_labeled.csv`.
3) (Revisão) Abrir `anotador_gui_erros.py` para ajustar rótulos e adicionar notas.
4) (Consumo) Dataset rotulado segue para modelagem (fora deste escopo).

### Convenções de colunas
- Identidade: `ticker`, `intervalo`, `estrategia_zigzag`, `tipo_padrao`
- Janelas: `data_inicio`, `data_fim`, `data_cabeca`, `data_retest`
- Métrica: `score_total`
- Rótulo: `label_humano` (0/1; -2 = ambíguo na revisão)

### Backlog priorizado (somente padrões)
1. Criar CLI leve para `necklineconfirmada.py` gerar `dataset_patterns_final.csv` (se ainda não existir entrypoint).
2. Unificar `ZIGZAG_STRATEGIES` e tolerâncias entre gerador e `anotador_gui.py` para consistência.
3. Normalizar datas (timezone-naive) em todo o pipeline e validar presença de colunas obrigatórias na carga.
4. Testes mínimos: cálculo de janela (zoom) e ZigZag; verificação de schema do CSV.
5. Documentar rapidamente o formato de entrada/saída na pasta `data/datasets/patterns_by_strategy/`.

### Riscos e notas
- Limites do Yahoo Finance para intervalos em minutos: usar períodos menores e reintentos (já contemplado nas GUIs).
- Datas com timezone: GUIs já convertem para naive; manter padrão no gerador.

### Comandos úteis
```bash
pip install -r requirements.txt
python src/patterns/analise/anotador_gui.py
python src/patterns/analise/anotador_gui_correto.py
python src/patterns/analise/anotador_gui_erros.py
```

### Testes
- Executar todos os testes (na raiz do projeto):
  - Windows/PowerShell: `python -m pytest -q`
  - Alternativo (Python launcher): `py -m pytest -q`
- Executar apenas o teste de DT/DB:
  - `python -m pytest -q tests/test_validate_double_pattern.py`


### Notas recentes
- DT/DB: adicionada chave `valid_neckline_retest_p4` (15) em `Config.SCORE_WEIGHTS_DTB`.
- DT/DB: `MINIMUM_SCORE_DTB` ajustado para 70 (antes 60).
- DT/DB: `identificar_padroes_double_top_bottom` agora usa janelas de 5 pivôs (inclui pivô de reteste `p4`) e repassa `p4` para validação.
- Correção: removido uso prematuro de `pivots_detectados`/`df_historico` no início de `main()` em `src/patterns/OCOs/necklineconfirmada.py`. A detecção de padrões agora é condicionada a `--patterns` apenas dentro do loop por ticker/estratégia/intervalo. Não houve mudança de interface de CLI.
- DT/DB: adicionado modo de depuração `Config.DTB_DEBUG` (default False). Ao ativar, imprime motivo de reprovação em cada regra obrigatória e confirmação de aceitação com `score` e `details`.
 - GUI (`anotador_gui.py`): `data_retest` agora mapeia `retest_p6_idx` para OCO/OCOI e `p4_idx` para DT/DB, permitindo destacar o reteste também nesses padrões.
