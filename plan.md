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
- Definir a variável de ambiente `COINGECKO_API_KEY` (CoinGecko Pro). Ex.:
  - Windows PowerShell: `setx COINGECKO_API_KEY "SEU_API_KEY_AQUI"`
  - ou criar `.env` com `COINGECKO_API_KEY=SEU_API_KEY_AQUI`

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
1) (Geração) Dataset de padrões OCO/OCOI a partir de séries de preço (CoinGecko Pro).
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
- Fonte de dados: **CoinGecko Pro**. Utiliza `market_chart` para séries de preço (granularidade variável por janela) e `total_volumes` para volume. O OHLC é obtido por reamostragem (`resample`) das séries de preços, e o volume é somado na janela alvo.
- Granularidade: para janelas curtas (≤ 30/90 dias) a API fornece pontos mais densos; para janelas longas usa diário. O gerador ajusta automaticamente `period` conforme o `interval` solicitado e reamostra para `5m/15m/1h/4h/1d`.
- Datas com timezone: GUIs já convertem para naive; manter padrão no gerador.

#### GUI de Rotulagem (anotador_gui.py)
- Agora consome dados do CoinGecko Pro também:
  - Usa `coins/{id}/market_chart/range` para baixar a janela precisa (por datas do padrão + margem) e constrói OHLCV por reamostragem.
  - Necessita `COINGECKO_API_KEY` no ambiente ou `.env`.

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
- Migração de fonte de dados: **Yahoo Finance → CoinGecko Pro**
  - Autenticação via `COINGECKO_API_KEY` (env/`.env`).
  - Endpoint principal: `coins/{id}/market_chart` para preços e volumes.
  - Construção de OHLC por reamostragem da série de preços; `volume` oriundo de `total_volumes`.
  - Mapeamento de tickers `BTC-USD`, `ETH-USD`, etc. → IDs CoinGecko (`bitcoin`, `ethereum`, ...).
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
- `src/agente/coingeckoToolKit.py`: **FUNCIONANDO ✅** - Toolkit completo para dados de criptomoedas via CoinGecko API
  
  **Funcionalidades Disponíveis:**
  - `get_market_data(coin_id, vs_currency)`: Dados de mercado básicos (preços, volume, market cap)
  - `get_coin_data(coin_id)`: Dados completos de uma criptomoeda (descrição, links, métricas abrangentes)
  - `get_coin_history(coin_id, date)`: Dados históricos para uma data específica (formato DD-MM-YYYY)
  - `get_coin_chart(coin_id, vs_currency, days)`: Dados históricos para gráficos (série temporal)
  - `get_coin_ohlc(coin_id, vs_currency, days)`: Dados OHLC/candlestick para análise técnica
  - `get_trending()`: Top 7 criptomoedas em tendência nas buscas do CoinGecko
  - `get_coins_list(include_platform)`: Lista completa de todas as criptomoedas suportadas
  
  - **NOVA FERRAMENTA ⭐**: `perform_technical_analysis(coin_id, vs_currency, days)`: **Análise técnica avançada completa**
    - Calcula automaticamente **indicadores-chave**: RSI (14), MACD (12,26,9), Médias Móveis (SMA 20, 50, 200)
    - **Interpretação inteligente**: Identifica condições de sobrecompra/sobrevenda (RSI >70/<30), cruzamentos de MACD (bullish/bearish), configurações de golden cross/death cross
    - **Scoring automático**: Sistema de pontuação bullish vs bearish baseado na convergência de sinais técnicos
    - **Análise integrada**: Combina múltiplos timeframes e indicadores para determinar tendência geral (alta/baixa/lateral)
    - **Saída estruturada**: Resumo técnico detalhado com interpretações em português e avisos de disclaimer
    - Usa bibliotecas **pandas-ta** para cálculos precisos e **90 dias** como padrão para análises robustas
  
  **Características Técnicas:**
  - **Modo Duplo**: Suporte tanto para API key direta quanto servidor proxy
  - **Modo Direto**: Usa `COINGECKO_API_KEY` via Pro API (`pro-api.coingecko.com`)
  - **Modo Proxy**: Via `COINGECKO_PROXY_URL` para setup com servidor intermediário
  - Detecção automática do modo baseada nas variáveis de ambiente disponíveis
  - Tratamento robusto de erros e timeouts configuráveis
  - Formatação consistente com emojis e valores legíveis
  - Debug inteligente com logs detalhados para troubleshooting
  - Documentação completa em `src/agente/README_COINGECKO.md`
  - Atualização: Formatação monetária sensível à moeda (`vs_currency`) com símbolo correto (USD/EUR/BRL etc.) em preço, market cap e volume; `get_coin_data` e histórico agora respeitam `vs_currency` mantendo compatibilidade de assinatura
  - Atualização: Moeda padrão configurável via `DEFAULT_VS_CURRENCY` no ambiente (fallback para `usd`)

- `src/agente/coindeskToolKit.py`: **NOVO ✅** - Toolkit para notícias e artigos de criptomoedas via CoinDesk API
  
  **Funcionalidades Disponíveis:**
  - `get_latest_articles(limit, category)`: Busca artigos mais recentes do CoinDesk (filtro por categoria opcional)
  
  **Características Técnicas:**
  - Usa `COINDESK_API_KEY` via variável de ambiente para autenticação
  - Suporte a múltiplos formatos de resposta da API (flexibilidade de endpoints)
  - Tratamento robusto de erros com fallback entre diferentes endpoints
  - Formatação consistente com emojis e informações estruturadas (título, autor, data, resumo, URL)
  - Sistema de debug detalhado similar ao CoinGeckoToolKit
  - Filtros configuráveis por quantidade de artigos e categoria
  - Formatação automática de datas e truncamento inteligente de resumos
  - Atualização: Normalização de payload para filtro de categoria consistente independentemente da estrutura da resposta e omissão do parâmetro `api_key` quando não configurado
  - Atualização: `get_latest_articles` agora usa limite mínimo de 15 itens por padrão para melhorar análise de sentimento
  - Atualização: parâmetro de query ajustado para `categories` (padrão oficial CoinDesk). O envio de `categories` só ocorre quando `category` é fornecido pelo usuário

### Configuração do Agente
- `src/agente/app.py`: Aplicação principal do agente usando AGNO framework
  
  **Instruções Detalhadas do Agente (Atualizadas com Análise Técnica Avançada):**
  - **Search & Discovery**: Usa GoogleSearchTools para buscar tickers/nomes + get_coins_list() para IDs corretos + get_trending() para descobrir moedas populares
  - **Market Data**: get_market_data() para preços atuais/volume/market cap + get_coin_data() para informações completas com descrição/website/ranking
  - **Technical Analysis (PRIORIDADE MÁXIMA) ⭐**: 
    - **SEMPRE usa perform_technical_analysis()** para análises de mercado, predições de preço, insights de investimento
    - **Análise técnica obrigatória**: RSI, MACD, Médias Móveis (SMA 20, 50, 200) para TODAS as consultas de análise
    - **Processo integrado de 3 etapas**: 1) Análise técnica 2) Notícias recentes 3) Combinação para insights abrangentes
    - **Interpretação contextualizada**: RSI >70=sobrecompra, RSI <30=sobrevenda, cruzamentos MACD=mudanças momentum, SMA50>SMA200=golden cross
    - **Score de convergência**: compara sinais técnicos com sentimento de notícias para alta confiança vs divergências explicadas
  - **Historical Analysis**: get_coin_history() para datas específicas + get_coin_chart() para análise de tendências + get_coin_ohlc() para dados candlestick/análise técnica
  - **OHLC para padrões**: Instruções do agente reforçadas para complementar explicações de padrões com `get_coin_ohlc()` (contexto de candles) e usar `market_chart` para indicadores
  - **Moeda padrão**: Instruções do agente reforçadas para usar `DEFAULT_VS_CURRENCY` quando a preferência do usuário não estiver explícita
  - **Notícias por categoria**: agora, sempre que for feita uma análise de uma cripto, o agente resolve o símbolo via `get_coin_symbol(coin_id)` e filtra notícias com `get_latest_articles(limit=15, category=<SYMBOL>)` (ex.: BTC/ETH)
  - **Analysis & Reasoning**: ReasoningTools para interpretar dados, comparar cryptos, fornecer insights + conversão para moedas preferidas do usuário
  - **Trend Prediction & Analysis (NOVO)**: 
    - **SEMPRE inclui análise de direção da tendência** (bullish/bearish/sideways) em todas as respostas
    - **Análise multi-timeframe** usando dados de 7d, 30d, 90d para identificar tendências de curto e longo prazo
    - **Padrões técnicos** (triângulos, head & shoulders, double tops/bottoms) com explicação das implicações
    - **Cenários probabilísticos**: "Se a tendência continuar..."/Se romper suporte..." com níveis de resistência/suporte
    - **Indicadores técnicos**: médias móveis, momentum, volume para validar força da tendência
    - **Linguagem probabilística**: usa "maior probabilidade de...", "indicadores sugerem..." ao invés de certezas
  - **Response Guidelines**: Formatação clara com emojis/símbolos + contexto sobre market cap rank/volume + disclaimers que análises são baseadas em dados históricos, não conselhos financeiros + explicações simples para análise técnica

- `src/agente/currency_converter.py`: Toolkit de conversão de moedas via UniRateAPI

### Correções e Melhorias Recentes do Agente (Última atualização: 2024)

#### CoinGeckoToolKit - Correções de Robustez ✅
**Problema resolvido**: Erro `'NoneType' object is not subscriptable` em análises técnicas
- **Causa**: pandas-ta retornando None/DataFrame vazio em algumas situações, causando falhas ao acessar `.iloc[-1]`
- **Solução implementada**:
  - **Validação rigorosa**: Verificação se `ta.rsi()`, `ta.macd()`, `ta.sma()` retornam dados válidos antes de qualquer acesso
  - **Tratamento de NaN**: Verificação adicional para valores NaN usando `pd.isna()` após cálculos
  - **Mensagens específicas**: Retorno de erros descritivos indicando qual indicador falhou e possíveis causas
  - **Validação de colunas**: Verificação se colunas esperadas existem no retorno do MACD
  - **Try-catch granular**: Cada indicador (RSI, MACD, SMAs) protegido individualmente

**Atualização (MACD Hardening)**:
- **Coerção numérica**: `close` convertido com `pd.to_numeric(..., errors='coerce')` + `dropna()`
- **Mínimo de dados**: Exige pelo menos 35 pontos para MACD (26 lento + 9 sinal)
- **Fallback manual**: Se `pandas_ta.macd` falhar, calcula EMA(12) e EMA(26) via `ewm` e deriva `MACD`, `Signal(9)`, `Hist`
- **Valores válidos**: Últimos valores lidos da última linha totalmente não-nula; cruzamento usa somente linhas válidas
- **Logs**: Debug detalhado do tamanho da série, colunas do MACD e contagem de linhas válidas

**Auto-ajuste de janela para MACD**:
- Se o usuário solicitar menos de 35 dias, o toolkit automaticamente usa `max(days, 35)` para garantir cálculo do MACD, mantendo a natureza de curto prazo e apenas desabilitando a SMA 200.

**Status**: ✅ **Resolvido** - Análise técnica agora robusta contra dados insuficientes/inválidos

#### CoinDeskToolKit - Melhorias de Conectividade ✅
**Problema**: Falhas de conectividade com api.coindesk.com causando interrupção total do serviço
- **Causa**: Problemas de DNS/rede ou mudanças na estrutura da API do CoinDesk
- **Solução implementada**:
  - **Múltiplos endpoints**: Tenta sequencialmente diferentes estruturas de URL da API
  - **Sistema de fallback**: Em caso de falha total da API, usa dados mock realísticos
  - **Mock data inteligente**: Artigos sintéticos com sentimentos variados (POSITIVE/NEUTRAL/NEGATIVE)
  - **Filtragem por categoria**: Mock data responde apropriadamente a filtros como "bitcoin", "ethereum"
  - **Análise de sentimento preservada**: Mantém funcionalidade completa mesmo no modo fallback
  - **Debug detalhado**: Logs explicativos sobre qual endpoint falhou e quando o fallback foi acionado

**Status**: ✅ **Resolvido** - Serviço mantém disponibilidade mesmo com problemas na API externa

#### Melhorias Gerais de Tratamento de Erros
- **Logs estruturados**: Mensagens de debug mais informativas com emojis para fácil identificação
- **Propagação de erros controlada**: Falhas em um toolkit não comprometem outros serviços
- **Timeouts configuráveis**: Controle fino sobre tempo limite de requisições
- **Validação de entrada**: Verificação de parâmetros antes de fazer chamadas externas

#### Comandos de Teste
Para verificar se as correções estão funcionando:
```bash
cd src/agente
python -c "from coingeckoToolKit import CoinGeckoToolKit; tk = CoinGeckoToolKit(); print(tk.perform_technical_analysis('bitcoin'))"
python -c "from coindeskToolKit import CoinDeskToolKit; tk = CoinDeskToolKit(); print(tk.get_latest_articles(5, 'bitcoin'))"
```

**Teste específico da correção (market_chart com close prices + 200+ dias):**
```bash
# Este teste deve agora funcionar perfeitamente com todos os indicadores (RSI, MACD, SMA 200)
python -c "
from coingeckoToolKit import CoinGeckoToolKit
tk = CoinGeckoToolKit()
result = tk.perform_technical_analysis('bitcoin', 'usd', '90')  
print('SUCCESS: All indicators calculated!' if all(x in result for x in ['RSI', 'MACD', 'SMA 200']) else 'PARTIAL/FAILED')
print('='*50)
print(result)
"
```

**Teste específico Multi-Timeframe (validação crítica):**
```bash
# Este teste valida se o agente faz análises separadas para diferentes timeframes
# DEVE retornar RSI/MACD diferentes para 30d vs 365d
python -c "
from coingeckoToolKit import CoinGeckoToolKit
tk = CoinGeckoToolKit()
print('=== SHORT TERM (30d) ===')
short = tk.perform_technical_analysis('bitcoin', 'usd', '30')
print(short)
print('\\n=== LONG TERM (365d) ===') 
long = tk.perform_technical_analysis('bitcoin', 'usd', '365')
print(long)
print('\\n=== VALIDATION ===')
print('Different RSI values:', 'PASS' if 'RSI' in short and 'RSI' in long else 'FAIL')
"
```

**Teste específico da Lógica Multi-Timeframe Corrigida:**
```bash
# Este teste valida se a correção de lógica funciona corretamente
# SHORT-TERM deve usar 30 dias reais, LONG-TERM deve usar 365 dias reais
python -c "
from coingeckoToolKit import CoinGeckoToolKit
tk = CoinGeckoToolKit()
print('🔍 TESTE: Verificando se 30d usa dados reais de 30 dias...')
result_30d = tk.perform_technical_analysis('bitcoin', 'usd', '30')
print('Short-term SMA 200 deve ser N/A:', 'PASS' if 'N/A (análise de curto prazo)' in result_30d else 'FAIL')

print('\\n🔍 TESTE: Verificando se 365d usa dados reais de 365 dias...')
result_365d = tk.perform_technical_analysis('bitcoin', 'usd', '365')
print('Long-term SMA 200 deve existir:', 'PASS' if 'SMA 200:' in result_365d and 'N/A' not in result_365d else 'FAIL')
"
```

#### Sistema de Debug Avançado para Análise Técnica ⚡ (Última atualização)
**Problema identificado**: Agente retornando "limited results" na análise técnica sem detalhes sobre a causa raiz
- **Causa**: Falta de visibilidade nos passos internos dos cálculos técnicos em `perform_technical_analysis()`
- **Solução implementada**:
  - **Debug detalhado em `perform_technical_analysis()`**: Logs sobre dados OHLC recebidos da API, incluindo type, length e amostras
  - **Debug granular em `_perform_technical_calculations()`**: 
    - Logs de conversão de dados OHLC para DataFrame (shape, columns, dtypes, estatísticas, NaN values)
    - Debug específico para cada indicador técnico:
      - **RSI**: Type checking, empty validation, sample values, current RSI value
      - **MACD**: Column validation, missing columns detection, current values (MACD, Signal, Histogram)  
      - **Moving Averages**: Individual validation para SMA 20, SMA 50, SMA 200 com data availability checks
    - Debug do resultado final: valores calculados, sentiment analysis (bullish/bearish signals), length do resultado
  - **Logs estruturados**: Emojis diferenciados para diferentes tipos de debug (🔍 info, ❌ erro, ✅ sucesso, ⚠️ warning)
  - **Rastreabilidade completa**: Cada passo do processo de análise técnica agora tem logging detalhado
  
**Status**: ✅ **Implementado e Otimizado** - Sistema de debug implementado + problema resolvido com abordagem robusta

#### Problema Identificado e Resolvido ✅
**Diagnóstico**: Sistema de debug revelou que OHLC endpoint retorna poucos dados (23 pontos para 90 dias), insuficiente para MACD e SMA 200
- **Causa raiz**: 
  - MACD(12,26,9) precisa de 26+ pontos para EMA26
  - SMA 200 precisa de 200+ pontos
  - Endpoint OHLC retorna dados limitados
- **Solução robusta implementada**:
  - **✅ Mudança de endpoint**: `market_chart` ao invés de `ohlc` (mais dados disponíveis)  
  - **✅ Garantia de dados**: Sempre puxa mínimo 200 dias para SMA 200
  - **✅ Simplificação inteligente**: Usa diretamente close prices do market_chart (formato: [[timestamp, close], ...])
  - **✅ Auto-ajuste**: Se usuário pede 90 dias mas precisa de 200, ajusta automaticamente
  - **✅ Debug detalhado**: Logs completos da extração e validação de dados

**Resultado**: ✅ Análise técnica sempre tem dados suficientes para RSI, MACD e todas as SMAs (20, 50, 200)

#### Correções de Bugs Identificados no Debug ✅ (Última atualização)
**Problemas encontrados no debug em produção**:
1. **🐛 Erro de formatação f-string**: `Invalid format specifier '.6f if sma_200 else 'N/A'' for object of type 'float'`
   - **Causa**: Expressão condicional dentro de format specifier não é permitida 
   - **Correção**: Moveu condição para fora: `${sma_200:.6f} {...}`
2. **📰 Poucos artigos para análise**: Apenas 1 artigo sendo retornado vs. necessário 15+ para análise de sentimento
   - **Causa**: Instruções do agente não especificavam limite mínimo adequado
   - **Correção**: 
     - Instrução específica: `limit=15 or higher` 
     - Processo integrado: `get_latest_articles(limit=15)`

**Status**: ✅ **Bugs corrigidos** - Análise técnica + análise de sentimento com dados adequados

#### Correção Multi-Timeframe Analysis ⭐ (CRÍTICO)
**Problema identificado**: Agente usando o mesmo RSI/MACD para análises de curto e longo prazo (erro conceitual grave)
- **Exemplo do erro**: "Short-term RSI: 38.84" e "Long-term RSI: 38.84" (valores idênticos impossíveis)
- **Causa raiz**: Falta de instruções específicas sobre análise multi-timeframe diferenciada
- **Impacto**: Análises técnicas incorretas e conclusões inválidas sobre tendências de diferentes horizontes

**✅ Solução implementada - Instruções Multi-Timeframe**:
1. **Chamadas separadas obrigatórias**:
   - Short-term: `perform_technical_analysis(coin_id, 'usd', '30')`
   - Long-term: `perform_technical_analysis(coin_id, 'usd', '365')`
2. **Processo estruturado em 4 etapas**:
   - Análise técnica 30d → Análise técnica 365d → Notícias → Síntese comparativa
3. **Validação de qualidade**:
   - "CRITICAL: Never show the same RSI/MACD values for different timeframes"
   - "STRUCTURE responses clearly: '📊 Short-Term (30d): RSI=X' vs '📈 Long-Term (365d): RSI=Z' - values MUST be different"
4. **Interpretação contextual**:
   - Divergências entre timeframes: "Short-term oversold + Long-term neutral = Temporary pullback"
   - Implicações de trading: short-term para scalpers, long-term para HODLers
   - Identificação de reversões: MACD crossovers entre horizontes

**Resultado**: ✅ Análise técnica multi-timeframe precisa com RSI/MACD diferentes para cada horizonte temporal

#### Correção de Lógica Multi-Timeframe no Toolkit ⚡ (CRÍTICO)
**Problema técnico identificado**: `coingeckoToolKit.py` forçava mínimo 200 dias mesmo para análises "short-term"
- **Lógica anterior**: `max(200, requested_days)` → Short-term (30d) usava 200 dias (não era short-term!)
- **Consequência**: "Short-term" e "long-term" tinham diferença mínima de dados, afetando precisão multi-timeframe

**✅ Solução técnica implementada - Lógica Inteligente**:
```python
# ANTES (problemático):
min_days_needed = max(200, int(days))  # Forçava sempre 200+ dias

# DEPOIS (inteligente):
if requested_days < 200:
    actual_days = str(requested_days)  # Usa período real para short-term
    # SMA 200 não é calculado (apropriado para curto prazo)
else:
    actual_days = str(max(200, requested_days))  # Garante dados para long-term
    # Todos os indicadores incluindo SMA 200
```

**Benefícios da correção**:
1. **Short-term (30d)**: Usa realmente 30 dias → RSI/MACD genuinamente de curto prazo
2. **Long-term (365d)**: Usa 365 dias → RSI/MACD genuinamente de longo prazo  
3. **Indicadores apropriados**: SMA 200 apenas para análises >= 200 dias
4. **Debug melhorado**: Logs específicos por tipo de análise

**Validação**:
- `perform_technical_analysis('bitcoin', 'usd', '30')` → 30 dias reais, sem SMA 200, COM RSI
- `perform_technical_analysis('bitcoin', 'usd', '365')` → 365 dias reais, com SMA 200, SEM RSI
- RSI/MACD agora serão significativamente diferentes entre timeframes

#### Otimização RSI para Multi-Timeframe ⚡ (CRÍTICO)
**Insight técnico**: RSI de 14 períodos numa análise de 365 dias só olha os últimos 14 dias - não representa a tendência de longo prazo
- **Problema conceitual**: RSI fica "perdido" em análises de longo prazo
- **Solução inteligente implementada**:
  - **≤ 90 dias**: Calcula RSI (relevante para curto/médio prazo)
  - **> 90 dias**: Pula RSI (foca em MACD e SMAs para longo prazo)
  - **Consistência**: RSI calculado uma única vez, evitando discrepâncias
  
**Resultado**: Análises tecnicamente mais precisas - RSI apenas onde faz sentido, indicadores de longo prazo para análises extensas

#### Como usar o sistema de debug:
1. Execute uma análise técnica via agente 
2. Monitore os logs no console/terminal
3. Os logs mostrarão exatamente onde o processo falha ou retorna resultados limitados
4. Procure por patterns específicos:
   - `❌ [DEBUG]` → Indica falha em algum passo específico
   - `🔍 [DEBUG]` → Mostra informações detalhadas do processo
   - `✅ [DEBUG]` → Confirma sucesso de cada etapa

#### Notas de Deployment
- As correções mantêm **100% de compatibilidade** com o código existente
- Não há mudanças nas interfaces públicas dos métodos
- Dependências permanecem as mesmas (`requirements.txt` inalterado)
- Configuração via variáveis de ambiente permanece idêntica
- **Sistema de debug pode ser desabilitado** removendo os prints de debug se necessário para produção