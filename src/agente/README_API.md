# 🚀 Crypto Analysis API & Interface

Esta aplicação oferece análise de criptomoedas através de uma API FastAPI e uma interface web interativa com Streamlit.

## 📋 Funcionalidades

- **🤖 Análise IA**: Análise completa de criptomoedas usando agente AI
- **📊 API REST**: Endpoints FastAPI para integração
- **🎨 Interface Web**: Dashboard interativo com Streamlit
- **😨 Fear & Greed Index**: Integração com índice de medo e ganância
- **📰 Análise de Sentimento**: Análise de notícias e sentimento do mercado
- **📈 Análise Técnica**: Indicadores técnicos (RSI, MACD, SMA)

## 🔧 Instalação

1. **Instale as dependências:**
```bash
pip install -r requirements_api.txt
```

2. **Configure as variáveis de ambiente:**
Certifique-se de que o arquivo `.env` contém:
```
OPENROUTER_API_KEY=your_openrouter_api_key_here
DEFAULT_VS_CURRENCY=usd
```

## 🚀 Como Usar

### Opção 1: Interface Completa (Recomendado)

**Passo 1: Inicie o servidor da API**
```bash
# Método 1: Usando o script helper
python src/agente/run_api.py

# Método 2: Diretamente
python src/agente/app.py serve
```

**Passo 2: Inicie a interface Streamlit (em outro terminal)**
```bash
# Método 1: Usando o script helper
python src/agente/run_streamlit.py

# Método 2: Diretamente
streamlit run src/agente/streamlit_interface.py
```

**Passo 3: Acesse a interface**
- **Streamlit Dashboard**: http://localhost:8501
- **API Documentation**: http://127.0.0.1:8000/docs

### Opção 2: Apenas API

```bash
# Inicie apenas a API
python src/agente/app.py serve --host 0.0.0.0 --port 8000
```

### Opção 3: CLI (Modo Original)

```bash
# Análise via linha de comando
python src/agente/app.py analyze bitcoin --term_type short
python src/agente/app.py analyze ethereum --term_type medium --vs_currency eur
```

## 📊 Endpoints da API

### `GET /`
Informações básicas da API

### `GET /health`
Status de saúde da API

### `GET /coins`
Lista de criptomoedas populares disponíveis

### `POST /analyze`
Realizar análise de criptomoeda

**Payload:**
```json
{
  "coin_id": "bitcoin",
  "vs_currency": "usd",
  "term_type": "short"
}
```

**Resposta:**
```json
{
  "ok": true,
  "data": {
    "type": "structured_crypto_analysis",
    "obtainable": {
      "price_current": 45000.0,
      "price_change_24h": 2.5,
      "volume_24h": 28500000000,
      "market_cap": 885000000000,
      "rsi": 65.2,
      "macd_signal": "bullish",
      "sma_20": 44500.0,
      "sma_50": 43000.0,
      "sma_200": 40000.0,
      "fear_greed_value": 68,
      "fear_greed_classification": "Greed"
    },
    "thoughts": {
      "summary": "Bitcoin shows strong bullish momentum with RSI indicating healthy buying pressure...",
      "price_trend": "bullish",
      "technical_signal": "buy",
      "resistance_levels": [46000, 48000],
      "support_levels": [44000, 42000],
      "news_sentiment": "positive",
      "market_sentiment": "optimistic",
      "investment_outlook": "bullish",
      "risk_level": "medium",
      "recommendation_confidence": 0.85,
      "key_factors": ["Strong technical indicators", "Positive market sentiment", "Institutional adoption"]
    },
    "metadata": {
      "coin_id": "bitcoin",
      "range": "30",
      "term_classification": "short"
    }
  },
  "errors": [],
  "meta": {
    "request_id": "..."
  }
}
```

## 🎯 Parâmetros de Análise

### Term Type (Tipo de Prazo)
- **short**: Análise de curto prazo (~30 dias) - Foco em momentum e day trading
- **medium**: Análise de médio prazo (~90 dias) - Equilibrio técnico/fundamental
- **long**: Análise de longo prazo (~365 dias) - Foco em fundamentos



## 🛠️ Desenvolvimento

### Estrutura de Arquivos
```
src/agente/
├── app.py                  # Aplicação principal com FastAPI
├── streamlit_interface.py  # Interface Streamlit
├── run_api.py             # Script para rodar API
├── run_streamlit.py       # Script para rodar Streamlit
└── README_API.md          # Esta documentação
```

### Testando a API

```bash
# Teste básico de saúde
curl http://127.0.0.1:8000/health

# Teste de análise
curl -X POST "http://127.0.0.1:8000/analyze" \
     -H "Content-Type: application/json" \
     -d '{"coin_id": "bitcoin", "vs_currency": "usd", "term_type": "short"}'
```

## 🎨 Interface Streamlit

A interface Streamlit oferece:

- **📊 Dashboard Visual**: Métricas em tempo real
- **⚙️ Configuração Fácil**: Sidebar com todos os parâmetros
- **📈 Visualizações**: Gráficos e indicadores visuais
- **🔍 Dados Detalhados**: Expandir para ver dados brutos
- **✅ Status da API**: Verificação automática de conectividade

## 🚨 Solução de Problemas

### API não inicia
- Verifique se todas as dependências estão instaladas
- Confirme que a variável `OPENROUTER_API_KEY` está configurada
- Verifique se a porta 8000 não está em uso

### Streamlit não conecta
- Certifique-se de que a API está rodando primeiro
- Verifique se a URL da API está correta no código do Streamlit
- Confirme que não há problemas de firewall

### Análise falha
- Verifique logs da API para erros específicos
- Confirme que o `coin_id` é válido no CoinGecko
- Verifique conectividade com internet para APIs externas

## 📝 Logs

Os logs da aplicação mostram:
- 🚀 Início de análises
- 📊 Chamadas de API para fontes de dados
- ⏱️ Tempo de execução
- ✅ Status de conclusão
- ❌ Erros e falhas

Para logs mais detalhados, monitore a saída do terminal onde a API está rodando.
