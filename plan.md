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
#### Robustez (ATR, ZigZag e validações)
- ATR: cálculo fortalecido em `calcular_indicadores` usando `append=False` e `squeeze()` para preencher `ATR_14` de forma confiável.
- Reteste por ATR (HNS/DTB/TTB): se `ATR_14` estiver indisponível ou for zero, aplica fallback de 0.5% do preço da neckline como tolerância mínima.
- ZigZag: extensão do último pivô agora exige desvio mínimo de 25% do `deviation_percent` configurado para evitar pivôs espúrios.
- `check_volume_profile`: validação de índices corrigida para cobrir `p1`, `p3` e `p5` (todos precisam ser >= 1).
- DT/DB: adicionada chave `valid_neckline_retest_p4` (15) em `Config.SCORE_WEIGHTS_DTB`.
- DT/DB: `MINIMUM_SCORE_DTB` ajustado para 70 (antes 60).
- DT/DB: `identificar_padroes_double_top_bottom` agora usa janelas de 5 pivôs (inclui pivô de reteste `p4`) e repassa `p4` para validação.
- Correção: removido uso prematuro de `pivots_detectados`/`df_historico` no início de `main()` em `src/patterns/OCOs/necklineconfirmada.py`. A detecção de padrões agora é condicionada a `--patterns` apenas dentro do loop por ticker/estratégia/intervalo. Não houve mudança de interface de CLI.
 - DT/DB: adicionado modo de depuração `Config.DTB_DEBUG` (default False). Ao ativar, imprime motivo de reprovação em cada regra obrigatória e confirmação de aceitação com `score` e `details`.
 - DT/DB: regra obrigatória `valid_contexto_extremos` — p1 e p3 devem ser extremos relevantes em janela baseada na distância média entre pivôs.
 - DT/DB: janela de contexto afrouxada: `HEAD_EXTREME_LOOKBACK_FACTOR=2` com piso `HEAD_EXTREME_LOOKBACK_MIN_BARS=30` barras.
 - DT/DB: adicionada regra obrigatória `valid_contexto_tendencia` (HH/HL para DT, LH/LL para DB) com tolerância mínima `DTB_TREND_MIN_DIFF_FACTOR=0.05` relativa à altura do padrão.
- DT/DB: logs de debug enriquecidos em `validate_and_score_double_pattern` reportando: tipos/preços dos pivôs ao falhar estrutura, janela de contexto/hi-lo ao falhar contexto, `min_sep` e preços ao falhar tendência, tolerância/altura/diff ao falhar simetria, e `ATR`, `mult`, `neckline`, `p4` e distância ao falhar reteste.
 - GUI (`anotador_gui.py`): `data_retest` agora mapeia `retest_p6_idx` para OCO/OCOI e `p4_idx` para DT/DB, permitindo destacar o reteste também nesses padrões.

#### Alteração: TTB contexto de extremos (p1) olhando apenas para trás
- Em `validate_and_score_triple_pattern`, a checagem `valid_contexto_extremos` do TTB passa a usar `is_head_extreme_past_only(...)`, que considera somente barras anteriores a `p1` (janela de tamanho idêntico à configuração vigente), em vez de uma janela centrada. O log de debug correspondente agora reporta `ctx_high/ctx_low` calculados somente no passado.

#### Refatorações recentes (necklineconfirmada.py)
- Debug padrão: `Config.DTB_DEBUG=False` e `Config.TTB_DEBUG=False` (desabilitado por padrão).
- Estocástico: adicionada `Config.STOCH_DIVERGENCE_REQUIRES_OBOS` para tornar opcional a exigência de OB/OS na divergência.
- is_head_extreme: agora exclui a própria barra do pivô da janela de contexto antes de calcular `max/min`, mantendo comparadores estritos (`>`/`<`) para garantir extremo único. Em caso de janela vazia após exclusão, falha fechada.
- find_breakout_index: breakout estrito (`>` para alta e `<` para baixa).
- detect_macd_signal_cross: regra flexibilizada — aceita cruzamento na direção correta dentro dos últimos `Config.MACD_CROSS_MAX_AGE_BARS` candles da janela; `MACD_CROSS_MAX_AGE_BARS=3` por padrão.
- validate_and_score_hns_pattern:
  - `valid_neckline_plana`: tolerância baseada na média das alturas dos dois ombros.
  - `valid_base_tendencia`: exige p0 estritamente abaixo (OCO) ou acima (OCOI) dos níveis da neckline (sem tolerância de 5%).
- Geração do CSV final:
  - Atribuições com máscara usando `.loc` em ambos os lados para `chave_idx`.
  - Preservação de todas as colunas: ordena `cols_info`, `cols_validacao`, `cols_pontos` e adiciona colunas restantes ao final.
#### Logging padrão
- Substituídos todos os `print(...)` por `logging` com configuração em `main()` para escrever em arquivo e console.
- Arquivo de log: `logs/run.log` (diretório criado automaticamente).
- Nível padrão: INFO (mudar via `logging.basicConfig` ou edição de código, se necessário).
- Debug unificado: `_pattern_debug(pattern_type, msg)` usa `logging.debug` e persiste mensagens sanitizadas por padrão em `Config.DEBUG_DIR`:
  - HNS → `hns_debug.log` (habilitado por `Config.HNS_DEBUG`)
  - DT/DB → `dtb_debug.log` (ou `Config.DTB_DEBUG_FILE` se definido; habilitado por `Config.DTB_DEBUG`)
  - TT/TB → `ttb_debug.log` (habilitado por `Config.TTB_DEBUG`)

#### ZigZag — desempate e parametrização
- Empates no mesmo índice em `calcular_zigzag_oficial`: tratamento explícito priorizando alternância (se último pivô foi `VALE`, um `PICO` no mesmo índice tem prioridade, e vice-versa). Em empate de tipo, mantém o mais extremo.
- Parametrização do desvio mínimo de extensão via `Config.ZIGZAG_EXTENSION_DEVIATION_FACTOR` (default `0.25`); substitui o valor fixo previamente codificado.

#### Epic 1: Indicadores modularizados (RF-001 a RF-004)
- Centralizado em `Config` thresholds/pesos: RSI/RSI forte, Stochastic (K/D/smooth, zonas), lookbacks de cruzamentos, volume de breakout (N e multiplicador), janela de busca de breakout.
- Adicionado `assess_rsi_divergence_strength(...)` com gating (>70/<30) e classificação de força (>80/<20 ou delta mínimo), usado em HNS e DTB. Mantida retrocompatibilidade de `check_rsi_divergence`.
- Adicionado `detect_macd_signal_cross(...)` (evento separado da divergência) e pontuações específicas em `SCORE_WEIGHTS_*`.
- Adicionado `check_stochastic_confirmation(...)` (divergência e cruzamento %K/%D), considerado apenas se partir de OB/OS.
- Adicionado `find_breakout_index(...)` + `check_breakout_volume(...)` para validar aumento de volume no candle de rompimento da neckline.
- `validate_and_score_hns_pattern` e `validate_and_score_double_pattern` passam a consumir os novos módulos, preservando o pipeline e colunas de saída.

#### Integração TT/TB (Topo/Fundo Triplo)
- Novas funções: `identificar_padroes_ttb(pivots)` e `validate_and_score_triple_pattern(pattern, df)` reutilizam regras/indicadores do DTB (RSI/forte, MACD divergência + signal cross, Estocástico, OBV, volume de rompimento, ATR para reteste).
- Pipeline (`main`): quando `--patterns` inclui `TTB` (ou `ALL`), detecta candidatos TT/TB e valida, anexando ao dataset final.
- CSV final: TT/TB não adiciona mais chaves redundantes; usa apenas `padrao_tipo` e `score_total`.
- CSV final: chave de unicidade agora usa `cabeca_idx` (HNS), `p3_idx` (DT/DB) e `p5_idx` (TT/TB).
- TT/TB: escopo de varredura harmonizado com `RECENT_PATTERNS_LOOKBACK_COUNT` (somente candidatos recentes).
- Logs/Debug: com `Config.TTB_DEBUG=True`, imprime motivos de reprovação/aceite e `score` para TT/TB (via `_pattern_debug`), similar ao DTB. Agora, ao falhar `valid_contexto_extremos` em TT/TB, o log inclui `lookback_bars`, `p1_preco`, `ctx_high` e `ctx_low`, igual ao DTB, facilitando diagnóstico de por que `p1` não foi extremo.

#### Performance e correções (alta prioridade)
- `calcular_indicadores(df)`: nova função que pré-calcula RSI (close/high/low), MACD (linha/sinal/hist), Estocástico, OBV e ATR uma vez por `df` e adiciona colunas no próprio DataFrame.
- `main()`: após `buscar_dados(...)`, o DataFrame é enriquecido por `calcular_indicadores(df_historico)` antes da detecção/validação.
- Funções de validação passam a ler colunas pré-calculadas, removendo recomputações internas de indicadores:
  - `assess_rsi_divergence_strength`: usa `RSI_{len}_CLOSE|HIGH|LOW` conforme a série fonte.
  - `detect_macd_signal_cross`: usa `MACD_*`/`MACDs_*` e varre toda a janela por qualquer cruzamento.
  - `check_macd_divergence`: usa `MACDh_*` pré-computado.
  - `check_stochastic_confirmation`: usa `STOCHk_*`/`STOCHd_*` pré-computados.
  - Regras de reteste (`ATR`) usam coluna `ATR_14` pré-computada.
- Benefícios: redução drástica de recomputações, melhoria de tempo de execução e consistência de resultados.

#### Atualização: GUI de anotação com suporte a TTB
- `src/patterns/analise/anotador_gui.py` agora reconhece padrões `TT`/`TB` gerados por `src/patterns/OCOs/necklineconfirmada.py`.
- Ajustes:
  - Novas regras `regras_map_ttb` exibidas no boletim: `valid_neckline_retest_p6`, `valid_neckline_plana`, `valid_simetria_extremos`, `valid_profundidade_vale_pico`, `valid_divergencia_rsi`.
  - `data_fim` e `data_retest` passam a considerar `p6_idx` para TT/TB.
  - Seleção dinâmica do conjunto de regras no painel (HNS, DTB ou TTB).
- Requisitos de colunas no CSV para TTB: `p0_idx`, `p3_idx`, `p5_idx`, `p6_idx`, além das flags `valid_*` usadas no boletim.

## Changelog (TTB/DTB/HNS tolerâncias) - ajuste de regras
- Aumentado `DTB_SYMMETRY_TOLERANCE_FACTOR` de 0.20 → 0.35 para reduzir reprovações por simetria em TT/TB.
- Reduzido `DTB_TREND_MIN_DIFF_FACTOR` de 0.02 → 0.01 para flexibilizar HL/LH mínimos em contexto de tendência (DTB/TTB).
- Diminuído `HEAD_EXTREME_LOOKBACK_MIN_BARS` de 10 → 8 para contexto de extremo (p1) considerar janelas menores.
- Aumentado `NECKLINE_RETEST_ATR_MULTIPLIER` de 4 → 5.0 para tolerância de reteste da neckline (HNS/DTB/TTB).
- Alterado fallback de tolerância quando ATR indisponível: 0,5% → 1,0% do preço da neckline (HNS p6, DTB p4, TTB p6).
- Desligado gate OB/OS do Estocástico (`STOCH_DIVERGENCE_REQUIRES_OBOS = False`) para ampliar confirmações opcionais.

Impacto esperado:
- Menos falsos negativos em TT/TB e DT/DB, mais candidatos aceitos quando estrutura é válida mas muito próxima dos limiares anteriores.
- Logs `ttb_debug.log` devem mostrar queda nas falhas por `valid_simetria_extremos`, `valid_contexto_extremos` e `valid_neckline_retest_p6`.

---

## Seção do Agente (fora do escopo principal)

### Toolkits AGNO
- `src/agente/coingeckoToolKit.py`: **FUNCIONANDO ✅** - Toolkit para dados de mercado de criptomoedas via CoinGecko API
  - Função principal: `get_market_data(coin_id, vs_currency)` para preços, volume e capitalização de mercado
  - **Modo Direto** (recomendado): Usa API key diretamente (`COINGECKO_API_KEY`) via Pro API (`pro-api.coingecko.com`)
  - **Modo Proxy** (avançado): Via servidor proxy que consulta endpoint `/coins/markets` (`COINGECKO_PROXY_URL`)
  - Detecção automática do modo baseada nas variáveis de ambiente disponíveis
  - Formato de saída formatado com emojis e valores legíveis
  - **Funcionalidades completas**: preços, mudança 24h, market cap, volume, sparkline
  - Debug inteligente com logs limpos e informativos
  - Documentação completa em `src/agente/README_COINGECKO.md`

### Configuração do Agente
- `src/agente/app.py`: Aplicação principal do agente usando AGNO framework
- `src/agente/currency_converter.py`: Toolkit de conversão de moedas via UniRateAPI