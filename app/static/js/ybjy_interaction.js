(function () {
    "use strict";

    function ready(callback) {
        if (document.readyState === "loading") {
            document.addEventListener("DOMContentLoaded", callback);
            return;
        }
        callback();
    }

    function formatFileCount(files) {
        if (!files || files.length === 0) {
            return "尚未选择文件";
        }
        if (files.length === 1) {
            return "已选择 1 个文件：" + files[0].name;
        }
        return "已选择 " + files.length + " 个文件";
    }

    function bindFileHint() {
        var input = document.querySelector("[data-ybjy-file-input]");
        var hint = document.querySelector("[data-ybjy-file-hint]");
        if (!input || !hint) {
            return;
        }
        input.addEventListener("change", function () {
            hint.textContent = formatFileCount(input.files);
        });
    }

    function clearPreviewStrip(strip) {
        while (strip.firstChild) {
            strip.removeChild(strip.firstChild);
        }
    }

    function createPreviewItem(file) {
        var item = document.createElement("div");
        var image = document.createElement("img");
        var name = document.createElement("span");

        item.className = "ybjy-preview-item";
        image.alt = file.name;
        name.textContent = file.name;

        item.appendChild(image);
        item.appendChild(name);

        if (window.URL && window.URL.createObjectURL) {
            image.src = window.URL.createObjectURL(file);
            image.addEventListener("load", function () {
                window.URL.revokeObjectURL(image.src);
            });
        }

        return item;
    }

    function bindLocalPreview() {
        var input = document.querySelector("[data-ybjy-file-input]");
        var strip = document.querySelector("[data-ybjy-preview-strip]");
        if (!input || !strip) {
            return;
        }

        input.addEventListener("change", function () {
            clearPreviewStrip(strip);
            Array.prototype.slice.call(input.files || [], 0, 8).forEach(function (file) {
                if (!file.type || file.type.indexOf("image/") !== 0) {
                    return;
                }
                strip.appendChild(createPreviewItem(file));
            });
        });
    }

    function bindConfirmForms() {
        var forms = document.querySelectorAll("[data-ybjy-confirm]");
        forms.forEach(function (form) {
            form.addEventListener("submit", function (event) {
                var message = form.getAttribute("data-ybjy-confirm") || "确认执行此操作吗？";
                if (!window.confirm(message)) {
                    event.preventDefault();
                }
            });
        });
    }

    function bindNumberInputs() {
        var inputs = document.querySelectorAll("[data-ybjy-number]");
        inputs.forEach(function (input) {
            input.addEventListener("change", function () {
                var min = parseInt(input.getAttribute("min") || "1", 10);
                var max = parseInt(input.getAttribute("max") || "50", 10);
                var value = parseInt(input.value || String(min), 10);
                if (Number.isNaN(value)) {
                    input.value = String(min);
                    return;
                }
                if (value < min) {
                    input.value = String(min);
                }
                if (value > max) {
                    input.value = String(max);
                }
            });
        });
    }

    function bindMessageAutoHide() {
        var messages = document.querySelectorAll(".ybjy-message");
        messages.forEach(function (message) {
            window.setTimeout(function () {
                message.style.transition = "opacity 0.3s ease";
                message.style.opacity = "0";
                window.setTimeout(function () {
                    if (message.parentNode) {
                        message.parentNode.removeChild(message);
                    }
                }, 320);
            }, 5600);
        });
    }

    function bindSubmitLock() {
        var forms = document.querySelectorAll("form");
        forms.forEach(function (form) {
            form.addEventListener("submit", function () {
                var button = form.querySelector("button[type='submit']");
                if (!button || button.hasAttribute("data-ybjy-no-lock")) {
                    return;
                }
                button.setAttribute("data-ybjy-original-text", button.textContent);
                window.setTimeout(function () {
                    button.disabled = true;
                    button.textContent = "处理中...";
                }, 0);
            });
        });
    }

    function normalizeText(text) {
        return String(text || "").toLowerCase().replace(/\s+/g, "");
    }

    function updateTableCount(table, countNode) {
        if (!table || !countNode) {
            return;
        }
        var rows = table.querySelectorAll("[data-ybjy-table-row]");
        var visibleCount = 0;
        rows.forEach(function (row) {
            if (row.style.display !== "none") {
                visibleCount += 1;
            }
        });
        countNode.textContent = "当前显示 " + visibleCount + " 条记录";
    }

    function bindTableFilter() {
        var input = document.querySelector("[data-ybjy-table-filter]");
        var table = document.querySelector("[data-ybjy-table]");
        var countNode = document.querySelector("[data-ybjy-table-count]");
        if (!input || !table) {
            return;
        }

        input.addEventListener("input", function () {
            var keyword = normalizeText(input.value);
            var rows = table.querySelectorAll("[data-ybjy-table-row]");
            rows.forEach(function (row) {
                var text = normalizeText(row.textContent);
                row.style.display = text.indexOf(keyword) >= 0 ? "" : "none";
            });
            updateTableCount(table, countNode);
        });
        updateTableCount(table, countNode);
    }

    function readCellValue(row, index) {
        var cells = row.querySelectorAll("td");
        if (!cells[index]) {
            return "";
        }
        return cells[index].textContent.trim();
    }

    function parseSortValue(value, type) {
        if (type === "number") {
            var numberValue = parseFloat(value);
            return Number.isNaN(numberValue) ? 0 : numberValue;
        }
        return normalizeText(value);
    }

    function compareSortValue(left, right, direction) {
        if (left < right) {
            return direction === "asc" ? -1 : 1;
        }
        if (left > right) {
            return direction === "asc" ? 1 : -1;
        }
        return 0;
    }

    function clearSortState(headers) {
        headers.forEach(function (header) {
            header.removeAttribute("data-ybjy-sort-active");
            header.removeAttribute("data-ybjy-sort-direction");
        });
    }

    function sortTableByColumn(table, columnIndex, type, direction) {
        var body = table.querySelector("tbody");
        if (!body) {
            return;
        }

        var rows = Array.prototype.slice.call(body.querySelectorAll("tr"));
        rows.sort(function (leftRow, rightRow) {
            var leftValue = parseSortValue(readCellValue(leftRow, columnIndex), type);
            var rightValue = parseSortValue(readCellValue(rightRow, columnIndex), type);
            return compareSortValue(leftValue, rightValue, direction);
        });

        rows.forEach(function (row) {
            body.appendChild(row);
        });
    }

    function bindTableSort() {
        var table = document.querySelector("[data-ybjy-table]");
        if (!table) {
            return;
        }

        var headers = Array.prototype.slice.call(table.querySelectorAll("[data-ybjy-sort]"));
        headers.forEach(function (header, index) {
            header.setAttribute("tabindex", "0");
            header.setAttribute("role", "button");
            header.addEventListener("click", function () {
                var currentDirection = header.getAttribute("data-ybjy-sort-direction");
                var nextDirection = currentDirection === "asc" ? "desc" : "asc";
                var type = header.getAttribute("data-ybjy-sort") || "text";
                clearSortState(headers);
                header.setAttribute("data-ybjy-sort-active", "1");
                header.setAttribute("data-ybjy-sort-direction", nextDirection);
                sortTableByColumn(table, index, type, nextDirection);
            });
            header.addEventListener("keydown", function (event) {
                if (event.key === "Enter" || event.key === " ") {
                    event.preventDefault();
                    header.click();
                }
            });
        });
    }

    function bindSearchResultAnchor() {
        var resultGrid = document.querySelector(".ybjy-result-grid");
        var hasResult = resultGrid && resultGrid.querySelector(".ybjy-result-item");
        if (!hasResult) {
            return;
        }
        if (window.location.hash) {
            return;
        }
        window.setTimeout(function () {
            resultGrid.scrollIntoView({ behavior: "smooth", block: "start" });
        }, 120);
    }

    function bindKeyboardShortcuts() {
        document.addEventListener("keydown", function (event) {
            if (event.key !== "Enter") {
                return;
            }
            var target = event.target;
            if (!target || target.tagName !== "INPUT") {
                return;
            }
            if (target.type === "search") {
                event.preventDefault();
            }
        });
    }

    function markLoadedImages() {
        var images = document.querySelectorAll(".ybjy-thumb");
        images.forEach(function (image) {
            if (image.complete) {
                image.classList.add("is-loaded");
                return;
            }
            image.addEventListener("load", function () {
                image.classList.add("is-loaded");
            });
        });
    }

    ready(function () {
        bindFileHint();
        bindLocalPreview();
        bindConfirmForms();
        bindNumberInputs();
        bindMessageAutoHide();
        bindSubmitLock();
        bindTableFilter();
        bindTableSort();
        bindSearchResultAnchor();
        bindKeyboardShortcuts();
        markLoadedImages();
    });
}());
