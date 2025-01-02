---

# WFiIS-AO-PROJEKT

**Aktualizacje na 02.01.2025:**

1. **Zaimplementowano wstępną obróbkę obrazów.**
2. **Dodano segmentację:**
   - Użyto wstępnie Pytesseract do analizy (na razie jako tymczasowe rozwiązanie, w przyszłości zostanie zastąpione przez nasz nauczony model).
3. **Zainstalowano dataset:** 
   - Link: [Letter Images Dataset](https://www.kaggle.com/datasets/kentvejrupmadsen/letter-images-dataset)
4. **Stworzono wstępny program do rozpoznawania znaków na obrazkach:**
   - Wczytano dane z folderu, gdzie obrazy są podzielone według etykiet, i przetworzono je do odpowiedniego formatu (odcienie szarości oraz przeskalowane do rozmiaru 30x50 pikseli).
   - Podzielono dane na zbiór treningowy i walidacyjny.
   - Wytrenowano model SVM do klasyfikacji obrazów (jest również próba robienia tego w PyTorch, jednak ta wersja jeszcze nie wyszła mi).
   - Sprawdzono dokładność modelu na danych testowych.
   - Zapisano wytrenowany model, aby móc go później wykorzystać do rozpoznawania znaków na nowych obrazkach.

**Uwagi:**
- Obecnie zastanawiamy się, czy potrzebujemy PyTorch. Na razie ogarnęłam temat bez niego, ale musimy się jeszcze dogadać, jakie zmiany są naprawdę potrzebne.
- Model nadal się uczy, co zajmuje sporo czasu z powodu dużego rozmiaru datasetu.

**Dalsze kroki:**
1. Uzgodnienie implementacji dalszych etapów projektu.
2. Analiza modelu do analizy znakow dla dobrej segmentacji.
3. Ktoś musi dodać kod przygotowujący obrazki rejestracji do rozpoznawania tablic rejestracyjnych (na końcowym etapie).
4. Wybór technologii:
   - Czy na pewno moje zmiany wymagają PyTorch? I czy chcemy konkretnie implementować na PyTorch?

---
