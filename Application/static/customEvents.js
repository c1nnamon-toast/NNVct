export function initializeCustomEvents(cy) {
    let lastTap = 0;
    let timeout;

    cy.on('tap', 'node', function(event) {
        let tappedNow = event.target;
        let currentTime = new Date().getTime();
        let tapInterval = currentTime - lastTap;
        
        if (tapInterval < 225 && tapInterval > 0) { // Check for double tap
            clearTimeout(timeout); // Prevent singleTap from firing
            tappedNow.emit('doubleTap', event); // Emit custom doubleTap event
            lastTap = 0; // Reset lastTap
        } else { // Single tap
            // Use a timeout to delay the singleTap action, allowing for a potential second tap
            timeout = setTimeout(() => {
                tappedNow.emit('singleTap', event); // Emit custom singleTap event if no subsequent tap occurs
            }, 225);
            lastTap = currentTime;
        }
    });
}
