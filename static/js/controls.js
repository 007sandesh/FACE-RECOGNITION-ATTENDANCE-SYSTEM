let isDragging = false;
let lastMouseX, lastMouseY;
let imageX = 0,
    imageY = 0;
let imageScale = 1;

function initControls() {
    canvas.addEventListener("mousedown", (event) => {
        if (!cameraEnabled && uploadedImage) {
            isDragging = true;
            lastMouseX = event.clientX;
            lastMouseY = event.clientY;
            canvas.style.cursor = "grabbing";
        }
    });

    canvas.addEventListener("mousemove", (event) => {
        if (isDragging) {
            const deltaX = event.clientX - lastMouseX;
            const deltaY = event.clientY - lastMouseY;
            imageX += deltaX;
            imageY += deltaY;
            lastMouseX = event.clientX;
            lastMouseY = event.clientY;
        }
    });

    canvas.addEventListener("mouseup", () => {
        isDragging = false;
        canvas.style.cursor = "default";
    });

    canvas.addEventListener("mouseleave", () => {
        isDragging = false;
        canvas.style.cursor = "default";
    });

    canvas.addEventListener("wheel", (event) => {
        if (!cameraEnabled && uploadedImage) {
            event.preventDefault();
            const zoomAmount = event.deltaY > 0 ? 0.9 : 1.1;
            const mouseX = event.clientX - canvas.offsetLeft;
            const mouseY = event.clientY - canvas.offsetTop;

            imageX = mouseX - (mouseX - imageX) * zoomAmount;
            imageY = mouseY - (mouseY - imageY) * zoomAmount;

            imageScale *= zoomAmount;
            imageScale = imageScale;
        }
    });
}