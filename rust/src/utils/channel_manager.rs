use crate::types::{Channel, MibiDescriptor};
use polars::prelude::*;

/// Create a DataFrame containing all channel information from a MibiDescriptor
pub fn extract_channels(descriptor: &MibiDescriptor) -> Result<DataFrame, PolarsError> {
    // Then try to extract from fov.panel
    if let Some(ref fov) = descriptor.fov {
        if let Some(panel) = &fov.panel {
            return to_dataframe(
                panel,
                descriptor.fov_mass_start(),
                descriptor.fov_mass_stop(),
            );
        }
    }

    // If we couldn't find any channels, return an error
    Err(PolarsError::NoData(
        "No channel information found in descriptor".into(),
    ))
}

// Helper function to create DataFrame from Panel
fn to_dataframe<T: AsRef<[Channel]>>(
    channels: T,
    mass_start: Option<f64>,
    mass_stop: Option<f64>,
) -> Result<DataFrame, PolarsError> {
    let channels = channels.as_ref();

    // Extract all channel properties into separate vectors
    let masses: Vec<Option<f64>> = channels.iter().map(|c| c.mass).collect();
    let targets: Vec<Option<String>> = channels.iter().map(|c| c.target.clone()).collect();
    let elements: Vec<Option<String>> = channels.iter().map(|c| c.element.clone()).collect();
    let clones: Vec<Option<String>> = channels.iter().map(|c| c.clone.clone()).collect();
    let mass_starts: Vec<Option<f64>> = channels
        .iter()
        .map(|c| match (c.mass, mass_start) {
            (Some(mass), Some(start)) => Some(mass - start.abs()),
            _ => None,
        })
        .collect();
    let mass_stops: Vec<Option<f64>> = channels
        .iter()
        .map(|c| match (c.mass, mass_stop) {
            (Some(mass), Some(stop)) => Some(mass + stop.abs()),
            _ => None,
        })
        .collect();
    let ids: Vec<Option<i32>> = channels.iter().map(|c| c.id).collect();
    let external_ids: Vec<Option<String>> =
        channels.iter().map(|c| c.external_id.clone()).collect();
    let concentrations: Vec<Option<f64>> = channels.iter().map(|c| c.concentration).collect();
    let lots: Vec<Option<String>> = channels.iter().map(|c| c.lot.clone()).collect();
    let manufacture_dates: Vec<Option<String>> = channels
        .iter()
        .map(|c| c.manufacture_date.clone())
        .collect();
    let conjugate_ids: Vec<Option<i32>> = channels.iter().map(|c| c.conjugate_id).collect();

    // Try to convert manufacture_dates to Date type if possible
    let date_series = Series::new("manufacture_date".into(), &manufacture_dates);
    let date_series = match date_series.cast(&DataType::Date) {
        Ok(ds) => ds,
        Err(_) => date_series, // Keep as string if conversion fails
    };

    // Create DataFrame with all columns
    DataFrame::new(vec![
        Series::new("target".into(), &targets).into(),
        Series::new("mass".into(), &masses).into(),
        Series::new("element".into(), &elements).into(),
        Series::new("clone".into(), &clones).into(),
        Series::new("mass_start".into(), &mass_starts).into(),
        Series::new("mass_stop".into(), &mass_stops).into(),
        Series::new("id".into(), &ids).into(),
        Series::new("external_id".into(), &external_ids).into(),
        Series::new("concentration".into(), &concentrations).into(),
        Series::new("lot".into(), &lots).into(),
        date_series.into(),
        Series::new("conjugate_id".into(), &conjugate_ids).into(),
    ])
}

/// Find channels by target name (case-insensitive)
pub fn find_by_target(df: &DataFrame, target: &str) -> Result<LazyFrame, PolarsError> {
    let target_lower = target.to_lowercase();

    // Perform a simple equality check with both lowercase and original target
    Ok(df.clone().lazy().filter(
        col("target")
            .eq(lit(target))
            .or(col("target").eq(lit(target_lower))),
    ))
}

/// Find a channel by mass with tolerance
pub fn find_by_mass(
    df: &DataFrame,
    mass: f64,
    tolerance: Option<f64>,
) -> Result<LazyFrame, PolarsError> {
    let tolerance = tolerance.unwrap_or(0.1);

    Ok(df.clone().lazy().filter(
        col("mass").is_not_null().and(
            col("mass")
                .gt_eq(lit(mass - tolerance))
                .and(col("mass").lt_eq(lit(mass + tolerance))),
        ),
    ))
}
