function onEdit(e) {
  var sheet = e.source.getSheetByName('Sheet1'); // Replace 'Sheet1' with the name of your sheet
  var range = e.range;
  var colIndex = range.getColumn();
  var colName = sheet.getRange(1, colIndex).getValue();

  // Check if the edited column is 'email'
  if (colName.toLowerCase() === 'email') {
    var data = sheet.getDataRange().getValues();
    var uniqueData = [];

    // Iterate through the data to find unique values based on email (case-insensitive)
    data.forEach(function(row) {
      var email = row[colIndex - 1].toString().toLowerCase(); // Convert to lowercase
      var index = uniqueData.findIndex(function(item) {
        return item[colIndex - 1].toString().toLowerCase() === email;
      });

      if (index !== -1) {
        // If duplicate, replace the existing entry with the latest one
        uniqueData[index] = row;
      } else {
        // If not a duplicate, add to the uniqueData array
        uniqueData.push(row);
      }
    });

    // Clear the entire sheet
    sheet.clearContents();

    // Write the unique data back to the sheet
    sheet.getRange(1, 1, uniqueData.length, uniqueData[0].length).setValues(uniqueData);
  }
}