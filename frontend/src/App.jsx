import React, { useState, useEffect } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, ReferenceLine, PieChart, Pie, Cell, BarChart, Bar } from 'recharts';
import { MapContainer, TileLayer, Marker, Popup, Circle } from 'react-leaflet';
import 'leaflet/dist/leaflet.css';
import L from 'leaflet';

// Fix leaflet icon missing issues
delete L.Icon.Default.prototype._getIconUrl;
L.Icon.Default.mergeOptions({
    iconRetinaUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-icon-2x.png',
    iconUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-icon.png',
    shadowUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-shadow.png',
});

import './index.css';

function App() {
  const [formData, setFormData] = useState({
    event_cause: 'vehicle_breakdown',
    priority: 'High',
    requires_road_closure: false,
    zone: 'Central Zone 2',
    hour_of_day: 18,
    day_of_week: 5,
    lat: 12.9716,
    lon: 77.5946
  });

  const [result, setResult] = useState(null);
  const [forecast, setForecast] = useState([]);
  const [health, setHealth] = useState({ model: 'XGBoost', mae: 0, drift_detected: false });
  const [globalImportances, setGlobalImportances] = useState([]);
  const [loading, setLoading] = useState(false);
  const API_URL =
  import.meta.env.VITE_API_URL ||
  "http://localhost:8000";
  
  useEffect(() => {
fetch(`${API_URL}/model-health`)
      .then(res => res.json())
      .then(data => setHealth(data))
      .catch(err => console.error(err));
      
    fetch(`${API_URL}/feature-importance`)
      .then(res => res.json())
      .then(data => setGlobalImportances(data.slice(0, 5)))
      .catch(err => console.error(err));
  }, []);

  useEffect(() => {
    const timer = setTimeout(() => {
      handleSubmit();
      fetchForecast();
    }, 300); // Fast debounce for instant what-if
    return () => clearTimeout(timer);
  }, [formData]);

  const handleChange = (e) => {
    const { name, value, type, checked } = e.target;
    setFormData(prev => ({
      ...prev, [name]: type === 'checkbox' ? checked : value
    }));
  };

  const fetchForecast = async () => {
    try {
      const response = await fetch(`${API_URL}/what-if`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ ...formData, hour_of_day: parseInt(formData.hour_of_day), day_of_week: parseInt(formData.day_of_week) })
      });
      setForecast(await response.json());
    } catch (e) {}
  };

  const handleSubmit = async (e) => {
    if (e) e.preventDefault();
    setLoading(true);
    try {
      const response = await fetch(`${API_URL}/generate-plan`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ ...formData, hour_of_day: parseInt(formData.hour_of_day), day_of_week: parseInt(formData.day_of_week) })
      });
      const data = await response.json();
      setResult(data);
    } catch (error) {} finally {
      setLoading(false);
    }
  };

  // Helper to calculate total cost
  const calculateCost = (res) => {
    if (!res) return 0;
    return (100*res.officers + 20*res.barricades + 300*res.patrol_vehicles + 500*res.ambulances + 400*res.tow_trucks + 1000*res.crane_units + 800*res.ert).toFixed(2);
  };

  return (
    <div style={{display: 'flex', flexDirection: 'column', height: '100vh'}}>
      <div className="header" style={{display: 'flex', justifyContent: 'space-between', alignItems: 'center'}}>
        <div className="header-left">
          <div className="tag">Operations Research Command Center (V11)</div>
          <h1>AI Traffic <span>Commander</span></h1>
          <div className="header-meta">Multi-Model Benchmark | PuLP Operations | Cosine KNN</div>
        </div>
        <div className="header-metrics" style={{display: 'flex', gap: '20px', background: 'var(--surface2)', padding: '15px 25px', borderRadius: '8px', border: '1px solid var(--border)'}}>
           <div><div style={{fontSize: '10px', color: 'var(--muted)', textTransform: 'uppercase'}}>Active Model</div><div style={{fontSize: '18px', fontWeight: 'bold'}}>{health.model}</div></div>
           <div><div style={{fontSize: '10px', color: 'var(--muted)', textTransform: 'uppercase'}}>Model MAE</div><div style={{fontSize: '18px', fontWeight: 'bold'}}>{health.mae}m</div></div>
           <div><div style={{fontSize: '10px', color: 'var(--muted)', textTransform: 'uppercase'}}>Drift Status (KS-Test)</div>
             <div style={{fontSize: '18px', fontWeight: 'bold', color: health.drift_detected ? 'var(--accent2)' : 'var(--accent3)'}}>
               {health.drift_detected ? 'DETECTED' : 'STABLE'}
             </div>
           </div>
        </div>
      </div>

      <div style={{display: 'flex', flex: 1, padding: '20px', gap: '20px', overflow: 'hidden'}}>
        
        {/* LEFT PANEL (25%) - WHAT-IF CONTROLS */}
        <div style={{width: '25%', background: 'var(--surface)', padding: '20px', borderRadius: '8px', overflowY: 'auto'}}>
            <div className="card-title"><div className="dot orange"></div>Scenario Controls</div>
            
            <div className="form-group" style={{marginTop: '20px'}}>
              <label>Event Cause</label>
              <select name="event_cause" value={formData.event_cause} onChange={handleChange} className="form-control">
                <option value="vehicle_breakdown">Vehicle Breakdown</option>
                <option value="accident">Accident</option>
                <option value="tree_fall">Tree Fall</option>
                <option value="water_logging">Water Logging</option>
                <option value="vip_movement">VIP Movement</option>
                <option value="public_event">Public Event / Festival</option>
                <option value="protest">Political Rally / Protest</option>
                <option value="construction">Construction</option>
              </select>
            </div>

            <div className="form-group">
               <label>Operational Zone</label>
               <select name="zone" value={formData.zone} onChange={handleChange} className="form-control">
                 <option value="Central Zone 1">Central Zone 1</option>
                 <option value="Central Zone 2">Central Zone 2</option>
                 <option value="Outer Zone">Outer Zone</option>
               </select>
            </div>

            <div className="form-group">
               <label>Priority</label>
               <select name="priority" value={formData.priority} onChange={handleChange} className="form-control">
                 <option value="High">High</option>
                 <option value="Low">Low</option>
               </select>
            </div>

            <div className="form-group" style={{background: 'rgba(255,255,255,0.02)', padding: '10px', borderRadius: '6px', border: '1px solid var(--border)'}}>
                <label>Requires Road Closure?</label>
                <div style={{marginTop: '10px'}}>
                   <input type="checkbox" name="requires_road_closure" checked={formData.requires_road_closure} onChange={handleChange} style={{transform: 'scale(1.5)', marginLeft: '5px'}} />
                </div>
            </div>

            <div className="form-group" style={{marginTop: '20px'}}>
               <label>Hour of Day</label>
               <select name="hour_of_day" value={formData.hour_of_day} onChange={handleChange} className="form-control">
                 {[...Array(24).keys()].map(h => (
                   <option key={h} value={h}>{h.toString().padStart(2, '0')}:00 {h >= 7 && h <= 9 || h >= 16 && h <= 19 ? '(Rush Hour)' : ''}</option>
                 ))}
               </select>
            </div>

            <div className="form-group">
               <label>Day of Week</label>
               <select name="day_of_week" value={formData.day_of_week} onChange={handleChange} className="form-control">
                 <option value="0">Monday</option>
                 <option value="1">Tuesday</option>
                 <option value="2">Wednesday</option>
                 <option value="3">Thursday</option>
                 <option value="4">Friday</option>
                 <option value="5">Saturday (Weekend)</option>
                 <option value="6">Sunday (Weekend)</option>
               </select>
            </div>
            
            {/* PUlP RESOURCE ALLOCATION MINI-VIEW */}
            {result && (
              <div style={{marginTop: '30px', borderTop: '1px solid var(--border)', paddingTop: '20px'}}>
                 <div className="card-title" style={{fontSize: '12px'}}><div className="dot red"></div>PuLP Deployment Strategy</div>
                 <div style={{display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '10px', marginTop: '15px'}}>
                    <div style={{background: 'var(--surface2)', padding: '10px', borderRadius: '6px', textAlign: 'center'}}>
                       <div style={{fontSize: '20px', fontWeight: 'bold'}}>{result.resources.officers}</div>
                       <div style={{fontSize: '10px', color: 'var(--muted)', textTransform: 'uppercase'}}>Officers</div>
                    </div>
                    <div style={{background: 'var(--surface2)', padding: '10px', borderRadius: '6px', textAlign: 'center'}}>
                       <div style={{fontSize: '20px', fontWeight: 'bold'}}>{result.resources.barricades}</div>
                       <div style={{fontSize: '10px', color: 'var(--muted)', textTransform: 'uppercase'}}>Barricades</div>
                    </div>
                    <div style={{background: 'var(--surface2)', padding: '10px', borderRadius: '6px', textAlign: 'center'}}>
                       <div style={{fontSize: '20px', fontWeight: 'bold'}}>{result.resources.patrol_vehicles}</div>
                       <div style={{fontSize: '10px', color: 'var(--muted)', textTransform: 'uppercase'}}>Patrol Veh</div>
                    </div>
                    <div style={{background: 'var(--surface2)', padding: '10px', borderRadius: '6px', textAlign: 'center'}}>
                       <div style={{fontSize: '20px', fontWeight: 'bold', color: result.resources.ambulances > 0 ? 'var(--accent2)' : 'var(--text)'}}>{result.resources.ambulances}</div>
                       <div style={{fontSize: '10px', color: 'var(--muted)', textTransform: 'uppercase'}}>Ambulances</div>
                    </div>
                    <div style={{background: 'var(--surface2)', padding: '10px', borderRadius: '6px', textAlign: 'center'}}>
                       <div style={{fontSize: '20px', fontWeight: 'bold', color: result.resources.tow_trucks > 0 ? 'var(--accent4)' : 'var(--text)'}}>{result.resources.tow_trucks}</div>
                       <div style={{fontSize: '10px', color: 'var(--muted)', textTransform: 'uppercase'}}>Tow Trucks</div>
                    </div>
                    <div style={{background: 'var(--surface2)', padding: '10px', borderRadius: '6px', textAlign: 'center'}}>
                       <div style={{fontSize: '20px', fontWeight: 'bold', color: result.resources.crane_units > 0 ? 'var(--accent)' : 'var(--text)'}}>{result.resources.crane_units}</div>
                       <div style={{fontSize: '10px', color: 'var(--muted)', textTransform: 'uppercase'}}>Cranes</div>
                    </div>
                    <div style={{background: 'var(--surface2)', padding: '10px', borderRadius: '6px', textAlign: 'center', gridColumn: '1 / -1'}}>
                       <div style={{fontSize: '20px', fontWeight: 'bold', color: result.resources.ert > 0 ? 'var(--accent3)' : 'var(--text)'}}>{result.resources.ert}</div>
                       <div style={{fontSize: '10px', color: 'var(--muted)', textTransform: 'uppercase'}}>Emergency Response Team</div>
                    </div>
                 </div>
              </div>
            )}

            {/* COUNTERFACTUAL EXPLANATION */}
            {result && result.counterfactual && (
               <div style={{marginTop: '20px', padding: '15px', background: 'rgba(56, 217, 169, 0.1)', border: '1px solid #38d9a9', borderRadius: '6px'}}>
                  <div style={{fontSize: '11px', color: '#38d9a9', fontWeight: 'bold', textTransform: 'uppercase', marginBottom: '5px'}}>Counterfactual Insight</div>
                  <div style={{fontSize: '13px', color: 'var(--text)'}}>{result.counterfactual}</div>
               </div>
            )}

        </div>

        {/* RIGHT PANEL (75%) - ANALYTICS */}
        <div style={{width: '75%', display: 'flex', flexDirection: 'column', gap: '20px', overflowY: 'auto', paddingRight: '10px'}}>
           
           {/* ROW 1: KPI CARDS */}
           {result && (
           <div style={{display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '15px'}}>
              <div style={{background: 'var(--surface)', padding: '20px', borderRadius: '8px', borderLeft: '4px solid var(--accent3)'}}>
                 <div style={{fontSize: '12px', color: 'var(--muted)', textTransform: 'uppercase', marginBottom: '10px'}}>Predicted Delay</div>
                 <div style={{fontSize: '36px', fontWeight: 'bold', color: 'var(--accent3)'}}>{result.expected_delay} <span style={{fontSize: '16px', color: 'var(--muted)'}}>min</span></div>
              </div>
              <div style={{background: 'var(--surface)', padding: '20px', borderRadius: '8px', borderLeft: '4px solid var(--accent)'}}>
                 <div style={{fontSize: '12px', color: 'var(--muted)', textTransform: 'uppercase', marginBottom: '10px'}}>95% Confidence (KNN)</div>
                 <div style={{fontSize: '28px', fontWeight: 'bold', color: 'var(--text)', marginTop: '8px'}}>{result.confidence_range[0]} - {result.confidence_range[1]}</div>
              </div>
              <div style={{background: 'var(--surface)', padding: '20px', borderRadius: '8px', borderLeft: '4px solid var(--accent2)'}}>
                 <div style={{fontSize: '12px', color: 'var(--muted)', textTransform: 'uppercase', marginBottom: '10px'}}>Operational Risk Score</div>
                 <div style={{fontSize: '36px', fontWeight: 'bold', color: result.risk_score > 70 ? 'var(--accent2)' : 'var(--accent)'}}>{result.risk_score} <span style={{fontSize: '16px', color: 'var(--muted)'}}>/ 100</span></div>
              </div>
              <div style={{background: 'var(--surface)', padding: '20px', borderRadius: '8px', borderLeft: '4px solid var(--accent4)'}}>
                 <div style={{fontSize: '12px', color: 'var(--muted)', textTransform: 'uppercase', marginBottom: '10px'}}>Deployment Cost Unit</div>
                 <div style={{fontSize: '36px', fontWeight: 'bold', color: 'var(--text)'}}>{calculateCost(result.resources)}</div>
              </div>
           </div>
           )}

           {/* ROW 2: GAUGE & KNN TABLE */}
           {result && (
           <div style={{display: 'grid', gridTemplateColumns: '1fr 2fr', gap: '20px'}}>
              <div className="card" style={{display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', padding: '30px 20px'}}>
                 <div className="card-title" style={{alignSelf: 'flex-start', marginBottom: '10px'}}><div className="dot red"></div>Risk Gauge</div>
                 {/* RECHARTS GAUGE */}
                 <ResponsiveContainer width="100%" height={150}>
                   <PieChart>
                     <Pie
                       data={[
                         { name: 'Risk', value: result.risk_score, fill: result.risk_score > 70 ? '#e84855' : (result.risk_score > 30 ? '#f5a623' : '#38d9a9') },
                         { name: 'Remaining', value: 100 - result.risk_score, fill: 'rgba(255,255,255,0.05)' }
                       ]}
                       cx="50%"
                       cy="100%"
                       startAngle={180}
                       endAngle={0}
                       innerRadius={60}
                       outerRadius={80}
                       dataKey="value"
                       stroke="none"
                     />
                     <text x="50%" y="90%" textAnchor="middle" fill="#ffffff" fontSize={36} fontWeight="bold">
                       {result.risk_score}
                     </text>
                   </PieChart>
                 </ResponsiveContainer>
              </div>
              
              <div className="card">
                 <div className="card-title"><div className="dot purple"></div>Similar Historical Incidents (Cosine KNN)</div>
                 <table style={{width: '100%', marginTop: '15px', borderCollapse: 'collapse', fontSize: '13px'}}>
                   <thead>
                     <tr style={{textAlign: 'left', color: 'var(--muted)', borderBottom: '1px solid var(--border)'}}>
                       <th style={{padding: '10px 5px'}}>Event Cause</th>
                       <th style={{padding: '10px 5px'}}>Zone</th>
                       <th style={{padding: '10px 5px'}}>Date</th>
                       <th style={{padding: '10px 5px', textAlign: 'right'}}>Observed Delay</th>
                       <th style={{padding: '10px 5px', textAlign: 'right'}}>Similarity</th>
                     </tr>
                   </thead>
                   <tbody>
                     {result.similar_events.map((ev, i) => (
                       <tr key={i} style={{borderBottom: '1px solid rgba(255,255,255,0.05)'}}>
                         <td style={{padding: '12px 5px'}}>{ev.cause}</td>
                         <td style={{padding: '12px 5px'}}>{ev.zone}</td>
                         <td style={{padding: '12px 5px', color: 'var(--muted2)'}}>{ev.date}</td>
                         <td style={{padding: '12px 5px', textAlign: 'right', fontWeight: 'bold', color: 'var(--accent)'}}>{ev.delay}m</td>
                         <td style={{padding: '12px 5px', textAlign: 'right'}}>
                            <span style={{background: 'rgba(56, 217, 169, 0.1)', color: '#38d9a9', padding: '4px 8px', borderRadius: '12px', fontSize: '11px', fontWeight: 'bold'}}>{ev.similarity}%</span>
                         </td>
                       </tr>
                     ))}
                   </tbody>
                 </table>
              </div>
           </div>
           )}

           {/* ROW 3: EXPLAINABILITY (SHAP & GLOBAL) */}
           {result && (
           <div style={{display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '20px'}}>
             <div className="card">
                <div className="card-title"><div className="dot green"></div>Local SHAP Contributions (This Event)</div>
                <div style={{height: '250px', width: '100%', marginTop: '10px'}}>
                   <ResponsiveContainer>
                     <BarChart data={result.shap.contributions} layout="vertical" margin={{top: 5, right: 30, left: 20, bottom: 5}}>
                       <XAxis type="number" stroke="#6b7897" />
                       <YAxis dataKey="feature" type="category" stroke="#6b7897" width={140} style={{fontSize: '11px'}} />
                       <Tooltip contentStyle={{background: '#0e1118', border: '1px solid #1e2535'}} />
                       <Bar dataKey="impact" radius={[0, 4, 4, 0]}>
                         {result.shap.contributions.map((entry, index) => (
                           <Cell key={`cell-${index}`} fill={entry.impact > 0 ? '#e84855' : '#38d9a9'} />
                         ))}
                       </Bar>
                     </BarChart>
                   </ResponsiveContainer>
                </div>
             </div>

             <div className="card">
                <div className="card-title"><div className="dot blue"></div>Global Model Importance ({health.model})</div>
                <div style={{height: '250px', width: '100%', marginTop: '10px'}}>
                   <ResponsiveContainer>
                     <BarChart data={globalImportances} layout="vertical" margin={{top: 5, right: 30, left: 20, bottom: 5}}>
                       <XAxis type="number" stroke="#6b7897" />
                       <YAxis dataKey="feature" type="category" stroke="#6b7897" width={140} style={{fontSize: '11px'}} />
                       <Tooltip contentStyle={{background: '#0e1118', border: '1px solid #1e2535'}} />
                       <Bar dataKey="importance" fill="#4d7eff" radius={[0, 4, 4, 0]} />
                     </BarChart>
                   </ResponsiveContainer>
                </div>
             </div>
           </div>
           )}

           {/* ROW 4: 24-HOUR WHAT-IF SIMULATION */}
           <div className="card" style={{marginBottom: '20px'}}>
             <div className="card-title" style={{display: 'flex', justifyContent: 'space-between'}}>
               <div><div className="dot purple"></div>24-Hour What-If Simulation</div>
               <div style={{fontSize: '12px', color: 'var(--muted)', fontWeight: 'normal'}}>Sensitivity Analysis Array</div>
             </div>
             <div style={{height: '250px', width: '100%', marginTop: '20px'}}>
                <ResponsiveContainer>
                  <LineChart data={forecast} margin={{top: 5, right: 20, left: 0, bottom: 15}}>
                    <XAxis dataKey="hour" stroke="#6b7897" tickFormatter={(t) => `${t}:00`} interval={0} angle={-45} textAnchor="end" tick={{fontSize: 10}} />
                    <YAxis stroke="#6b7897" domain={[0, 100]} tick={{fontSize: 10}} />
                    <Tooltip contentStyle={{background: '#0e1118', border: '1px solid #1e2535'}} />
                    <Line type="monotone" dataKey="risk" stroke="#f5a623" strokeWidth={3} dot={{r: 4, fill: '#f5a623'}} activeDot={{r: 8}} />
                  </LineChart>
                </ResponsiveContainer>
             </div>
           </div>

        </div>
      </div>
      <div style={{textAlign: 'center', padding: '20px', color: 'var(--muted)', fontSize: '13px', borderTop: '1px solid var(--border)', marginTop: '20px'}}>
        © 2026 Developed by Arpit Purohit. All rights reserved.
      </div>
    </div>
  );
}

export default App;
