# CoinGecko Toolkit - Configuração

## 📋 Resumo

O CoinGecko Toolkit suporta **duas formas** de configuração:

### 🎯 Opção 1: Modo Direto (RECOMENDADO)
- **Mais simples**: Não precisa de servidor proxy
- **Configura apenas**: API key da CoinGecko
- **Chama direto**: API da CoinGecko com autenticação

### 🔒 Opção 2: Modo Proxy (AVANÇADO) 
- **Mais seguro**: API key fica no servidor proxy
- **Requer**: Implementar servidor proxy
- **Configura**: URL do proxy server

---

## ⚙️ Como Configurar

### 1️⃣ Obter API Key da CoinGecko

1. Acesse: https://www.coingecko.com/pt/api/pricing
2. Faça login/cadastro
3. Clique em **"+ Adicionar Nova Chave"**
4. Copie sua API key

**⚠️ IMPORTANTE:** Este toolkit usa automaticamente a **Pro API** (`pro-api.coingecko.com`) que é necessária para API keys pagas. Se você estiver usando a versão gratuita, poderá ter limitações.

**✅ FUNCIONALIDADES COMPLETAS RESTAURADAS:**
- 📊 Mudança de preço em 24h 
- 📈 Gráfico sparkline (desabilitado para performance)
- 🌐 Headers HTTP completos (User-Agent, Accept)
- 🔧 Debug limpo e informativo

### 2️⃣ Configurar Variáveis de Ambiente

Crie um arquivo `.env` na raiz do projeto:

```bash
# Modo Direto (RECOMENDADO)
COINGECKO_API_KEY=your_coingecko_api_key_here

# OU Modo Proxy (OPCIONAL)
# COINGECKO_PROXY_URL=http://localhost:8000
```

---

## 🚀 Como Usar

### No seu agente (`src/agente/app.py`):

```python
from coingeckoToolKit import CoinGeckoToolKit

# Adicione ao seu agente
reasoning_agent = Agent(
    model=Gemini(id="gemini-1.5-flash-latest"),
    tools=[
        ReasoningTools(add_instructions=True),
        YFinanceTools(...),
        GoogleSearchTools(),
        CurrencyConverterTools(),
        CoinGeckoToolKit()  # ← ADICIONE ESTA LINHA
    ],
    # ... resto da configuração
)
```

### Exemplo de uso:

```python
toolkit = CoinGeckoToolKit()

# Obter dados do Bitcoin em USD
result = toolkit.get_market_data("bitcoin", "usd")

# Obter dados do Ethereum em BRL  
result = toolkit.get_market_data("ethereum", "brl")

# Obter dados do Cardano em EUR
result = toolkit.get_market_data("cardano", "eur")
```

---

## 🔧 Modo Proxy (Opcional)

Se preferir usar o modo proxy, você precisa implementar um servidor que:

1. **Receba** requisições do toolkit
2. **Adicione** a API key no header `x-cg-pro-api-key`
3. **Encaminhe** para a CoinGecko API
4. **Retorne** a resposta

### Exemplo de servidor proxy (Node.js/Express):

```javascript
const express = require('express');
const axios = require('axios');
const app = express();

app.get('/api/coingecko/markets', async (req, res) => {
  try {
    const response = await axios.get('https://api.coingecko.com/api/v3/coins/markets', {
      params: req.query,
      headers: {
        'x-cg-pro-api-key': process.env.COINGECKO_API_KEY
      }
    });
    res.json(response.data);
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

app.listen(8000, () => {
  console.log('Proxy server running on port 8000');
});
```

---

## 🎯 Detecção Automática

O toolkit detecta automaticamente qual modo usar:

- ✅ **Se `COINGECKO_API_KEY` estiver definida** → Modo Direto
- ✅ **Se `COINGECKO_PROXY_URL` estiver definida** → Modo Proxy
- ❌ **Se nenhuma estiver definida** → Erro de configuração

---

## 🪙 IDs de Moedas Comuns

Use estes IDs na função `get_market_data()`:

| Moeda | ID CoinGecko |
|-------|--------------|
| Bitcoin | `bitcoin` |
| Ethereum | `ethereum` |
| Cardano | `cardano` |
| Solana | `solana` |
| Polkadot | `polkadot` |
| Chainlink | `chainlink` |
| Polygon | `matic-network` |
| Avalanche | `avalanche-2` |

**💡 Dica**: Para encontrar o ID de qualquer moeda, acesse a página da moeda no CoinGecko e veja a URL. Ex: `https://www.coingecko.com/pt/coins/bitcoin` → ID é `bitcoin`

---

## ❓ Troubleshooting

### ❌ "Missing COINGECKO_API_KEY environment variable"
**Solução**: Configure a variável `COINGECKO_API_KEY` no seu `.env`

### ❌ "Error fetching market data"
**Possíveis causas**:
- API key inválida
- Limite de requisições atingido
- Problemas de conectividade
- ID da moeda incorreto

### ❌ Servidor proxy não responde
**Soluções**:
- Verifique se o servidor proxy está rodando
- Confirme a URL em `COINGECKO_PROXY_URL`
- Teste o endpoint do proxy manualmente
