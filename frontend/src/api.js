/**
 * Bati Bank Credit Risk API Client
 */

const API_BASE_URL = 'http://localhost:8000';

export async function checkHealth() {
  try {
    const res = await fetch(`${API_BASE_URL}/health`);
    if (!res.ok) throw new Error('API degraded');
    return await res.json();
  } catch (error) {
    return { status: 'offline', error: error.message };
  }
}

export async function scoreCustomer(customerId, features) {
  try {
    const res = await fetch(`${API_BASE_URL}/predict`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        customer_id: customerId,
        features: features
      })
    });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    return await res.json();
  } catch (error) {
    console.warn('API call failed, fallback to local scoring:', error);
    // Local fallback calculation matching DecisionTree / RFM boundaries
    const probHigh = (features.FraudRate > 0.10 || features.Recency > 30 || features.TotalAmount < 5000)
      ? 0.92
      : (features.FraudRate > 0.03 || features.Recency > 12 ? 0.45 : 0.02);

    const creditScore = Math.round(300 + (1 - probHigh) * 550);
    const decision = probHigh > 0.60 ? 'Decline' : (probHigh > 0.30 ? 'Review' : 'Approve');
    const maxAmount = probHigh > 0.60 ? 0 : (probHigh > 0.30 ? 5000 : 10000);

    return {
      success: true,
      prediction: {
        customer_id: customerId,
        predicted_class: probHigh > 0.5 ? 1 : 0,
        risk_probability: probHigh,
        confidence: Math.max(probHigh, 1 - probHigh),
        risk_level: probHigh > 0.6 ? 'High' : (probHigh > 0.3 ? 'Medium' : 'Low'),
        credit_score: creditScore
      },
      decision: {
        customer_id: customerId,
        decision: decision,
        risk_probability: probHigh,
        credit_score: creditScore,
        recommended_terms: {
          max_amount: maxAmount,
          max_duration_months: probHigh > 0.3 ? 6 : 12,
          interest_rate: probHigh > 0.3 ? 0.18 : 0.12
        }
      },
      model_version: 'DecisionTree-local-fallback'
    };
  }
}
