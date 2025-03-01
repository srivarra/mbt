use crate::types::{Channel, MibiDescriptor};
use std::collections::HashMap;

/// A utility for managing and looking up channels in MIBI data
pub struct ChannelManager {
    channels: Vec<Channel>,
    name_lookup: HashMap<String, usize>,
    mass_lookup: HashMap<u32, usize>, // We'll use quantized mass values (mass * 100) as keys
}

impl ChannelManager {
    /// Create a new ChannelManager from a MibiDescriptor
    pub fn new(descriptor: &MibiDescriptor) -> Self {
        // First try to get channels from top-level panel field
        let mut channels = Vec::new();

        if let Some(panel_channels) = descriptor.panel.as_ref() {
            channels = panel_channels.clone();
            println!("Found {} channels in top-level panel", channels.len());
        }
        // If no top-level panel, try to get from fov.panel.conjugates if available
        else if let Some(fov) = &descriptor.fov {
            // Access fov.panel if it exists in the JSON
            if let Some(fov_panel) = fov.extra.get("panel") {
                // Try to extract a "conjugates" array from the panel
                if let Some(conjugates) = fov_panel.get("conjugates") {
                    if let Some(conjugates_array) = conjugates.as_array() {
                        println!("Found {} potential channels in fov.panel.conjugates", conjugates_array.len());

                        // Convert each conjugate to a Channel
                        for conjugate in conjugates_array {
                            if let Some(obj) = conjugate.as_object() {
                                // Use target as the name (fallback to element if not available)
                                let name = obj.get("target")
                                    .and_then(|n| n.as_str())
                                    .or_else(|| obj.get("element").and_then(|e| e.as_str()))
                                    .unwrap_or("Unknown")
                                    .to_string();

                                let mass = obj.get("mass")
                                    .and_then(|m| m.as_f64());

                                let target = obj.get("target")
                                    .and_then(|t| t.as_str())
                                    .map(String::from);

                                let element = obj.get("element")
                                    .and_then(|e| e.as_str())
                                    .map(String::from);

                                let clone = obj.get("clone")
                                    .and_then(|c| c.as_str())
                                    .map(String::from);

                                let mass_start = obj.get("massStart")
                                    .and_then(|m| m.as_f64());

                                let mass_stop = obj.get("massStop")
                                    .and_then(|m| m.as_f64());

                                let id = obj.get("id")
                                    .and_then(|i| i.as_i64())
                                    .map(|i| i as i32);

                                let external_id = obj.get("external_id")
                                    .and_then(|e| e.as_str())
                                    .map(String::from);

                                let concentration = obj.get("concentration")
                                    .and_then(|c| c.as_f64());

                                let lot = obj.get("lot")
                                    .and_then(|l| l.as_str())
                                    .map(String::from);

                                let manufacture_date = obj.get("manufacture_date")
                                    .and_then(|m| m.as_str())
                                    .map(String::from);

                                let conjugate_id = obj.get("conjugate_id")
                                    .and_then(|c| c.as_i64())
                                    .map(|c| c as i32);

                                // Create extra map for any additional fields
                                let mut extra = HashMap::new();
                                for (key, value) in obj {
                                    if !["target", "element", "mass", "massStart", "massStop",
                                         "clone", "id", "external_id", "concentration", "lot",
                                         "manufacture_date", "conjugate_id"].contains(&key.as_str()) {
                                        extra.insert(key.clone(), value.clone());
                                    }
                                }

                                // Create a Channel from the extracted data
                                let channel = Channel {
                                    name,
                                    mass,
                                    target,
                                    element,
                                    clone,
                                    mass_start,
                                    mass_stop,
                                    id,
                                    external_id,
                                    concentration,
                                    lot,
                                    manufacture_date,
                                    conjugate_id,
                                    extra,
                                };

                                channels.push(channel);
                            }
                        }

                        println!("Successfully extracted {} channels from fov.panel.conjugates", channels.len());
                    }
                }
            }
        }

        let mut name_lookup = HashMap::new();
        let mut mass_lookup = HashMap::new();

        // Build lookup tables
        for (idx, channel) in channels.iter().enumerate() {
            // Add name-based lookup
            name_lookup.insert(channel.name.to_lowercase(), idx);

            // Add mass-based lookup (quantized for easier fuzzy matching)
            if let Some(mass) = channel.mass {
                // Quantize mass to integer by multiplying by 100
                let quantized_mass = (mass * 100.0).round() as u32;
                mass_lookup.insert(quantized_mass, idx);
            }
        }

        Self {
            channels,
            name_lookup,
            mass_lookup,
        }
    }

    /// Find a channel by name (case-insensitive)
    pub fn find_by_name(&self, name: &str) -> Option<&Channel> {
        let name_lower = name.to_lowercase();
        self.name_lookup.get(&name_lower).map(|&idx| &self.channels[idx])
    }

    /// Find a channel by mass with an optional tolerance
    pub fn find_by_mass(&self, mass: f64, tolerance: Option<f64>) -> Option<&Channel> {
        let tolerance = tolerance.unwrap_or(0.1); // Default tolerance of 0.1 Da
        let mass_int = (mass * 100.0).round() as u32;

        // Try exact match first
        if let Some(&idx) = self.mass_lookup.get(&mass_int) {
            return Some(&self.channels[idx]);
        }

        // If no exact match, try within tolerance
        let tolerance_int = (tolerance * 100.0).round() as u32;
        let lower_bound = mass_int.saturating_sub(tolerance_int);
        let upper_bound = mass_int.saturating_add(tolerance_int);

        for m in lower_bound..=upper_bound {
            if let Some(&idx) = self.mass_lookup.get(&m) {
                return Some(&self.channels[idx]);
            }
        }

        None
    }

    /// Find a channel by name or mass
    pub fn find_channel(&self, identifier: &str) -> Option<&Channel> {
        // Try to find by name first
        if let Some(channel) = self.find_by_name(identifier) {
            return Some(channel);
        }

        // If identifier looks like a numeric value, try to parse as mass
        if let Ok(mass) = identifier.parse::<f64>() {
            return self.find_by_mass(mass, None);
        }

        None
    }

    /// Get all available channel names
    pub fn get_channel_names(&self) -> Vec<String> {
        self.channels
            .iter()
            .map(|c| c.name.clone())
            .collect()
    }

    /// Get all channel masses
    pub fn get_channel_masses(&self) -> Vec<f64> {
        self.channels
            .iter()
            .filter_map(|c| c.mass)
            .collect()
    }

    /// Get all channels
    pub fn get_channels(&self) -> &[Channel] {
        &self.channels
    }
}
