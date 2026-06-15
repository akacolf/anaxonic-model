import unittest

from anaxonic.tokenizer import ByteTokenizer


class ByteTokenizerTest(unittest.TestCase):
    def test_byte_tokenizer_round_trips_unicode(self):
        tokenizer = ByteTokenizer()
        text = "Anaxonic: hello, dunya"
        self.assertEqual(tokenizer.decode(tokenizer.encode(text)), text)

    def test_byte_tokenizer_vocab_is_fixed(self):
        tokenizer = ByteTokenizer()
        self.assertEqual(tokenizer.vocab_size, 256)
        self.assertEqual(tokenizer.encode("A"), [65])


if __name__ == "__main__":
    unittest.main()
