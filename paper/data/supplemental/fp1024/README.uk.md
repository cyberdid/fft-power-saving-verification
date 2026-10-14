# Окремий потоковий F_p-тест при r=1024

Цей експеримент не входить до зафіксованого 135-крокового core-прогону. Його результати не додано заднім числом до core manifest/raw. Основний checker — `paper/scripts/stream_exact_dft.py`, версія checker v2 (поле `checker_version` у JSON); SHA256 9eeedcbaf0568ba2899d016a8a82a7c2950dd74924b45cf94225ebeb3b6fee3b.

Два ті самі цілочисельні вектори у двох полях, 4096 координат, 0 розбіжностей; 431,9 с і 37,91 MiB peak RSS за GNU time. Це необхідна умова на вибраних векторах, не доказ комплексної матриці. C-слоти r=1024 не перевірялись. Первинний прогін дав ті самі виходи/потоки за 450,95 с і 37,48 MiB. У первинному checker був хибний reverse diagnostic на no-reverse control; v2 виправляє діагностику фактично емітованого порядку. Основні арифметичні потоки та виходи первинного позитивного прогону збігаються з v2.

Із кореня проєкту, у середовищі release/requirements-lock.txt:

```bash
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python paper/scripts/stream_exact_dft.py 1024 --primes 2 --nvec 2 --json /path/to/new-result.json
```

Для r=1024 не використовуйте --materialise-check або --corrupt: вони матеріалізують слово. Фактичний v2-запуск мав virtual-memory cap 8000000 KiB. Для швидкої перевірки методу запустіть r=16 із --primes 1 --basis --materialise-check; очікується 0 розбіжностей і всі три materialised flags true. Контроли r=16 --basis --primes 1: --bad-transpose noreverse дає 224 розбіжності й reverse_equals_reversed_forward=false; --bad-transpose noswap — 255 і transpose_swaps_shear_ends=false; --corrupt — 60. Негативні контроли навмисно повертають exit0: перевіряйте JSON, а не лише код завершення.

Незалежна звірка з первинним результатом і execution-time source hashes: independent-check.json, 51 PASS. JSON містить історичні абсолютні шляхи машини виконання; для перенесеної копії зіставляйте вміст за SHA, а не вимагайте тих самих шляхів. Хеші джерел на диску не є атестацією bytecode або всіх runtime inputs. Gate cache перевірено окремо core-прогоном. Хеш порядку — fingerprint, не математичний доказ.

Поля maxrss_mb у JSON та MB у логу обчислено як ru_maxrss/1024, тобто це MiB; для статті використано GNU time. Оригінальні сирі файли не переписано.
