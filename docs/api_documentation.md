# API Documentation

## Endpoint
`POST /predict`

## Request
**Content-Type**: `application/json`

```json
{
  "image": "base64_encoded_string..."
}
```

## Response
**Content-Type**: `application/json`

```json
{
  "prediction": [0.95, 0.05], // [Probability Normal, Probability Anomaly]
  "confidence": 0.95,
  "is_anomaly": false
}
```

## Error Handling
- **400 Bad Request**: Missing image data.
- **500 Internal Server Error**: Inference failure.
