CDF       
      obs    8   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type                     	amplitude                   comment       �Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM���   
add_offset               min       ?`bM���   max       ?ʟ�vȴ:      �  �   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       NV\   max       P��      �  �   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       �y�#   max       >&�y      �  l   effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>nz�G�   max       @E��
=p�     �   L   effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ?��G�|    max       @v�fffff     �  )   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @'         max       @Q            p  1�   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @�        max       A �          �  2<   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       �P�`   max       >t�j      �  3   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A���   max       B*�n      �  3�   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A���   max       B*ߥ      �  4�   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       ?�   max       C���      �  5�   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       ?��   max       C���      �  6�   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max          �      �  7|   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min             max          W      �  8\   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min             max          S      �  9<   
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       NV\   max       P���      �  :   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6��C-   
add_offset               min       ?�����   max       ?��5�Xy>      �  :�   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       �y�#   max       >&�y      �  ;�   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>������   max       @ExQ��     �  <�   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ?��G�|    max       @v�fffff     �  E|   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @         max       @P�           p  N<   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @�        max       @�R�          �  N�   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         E�   max         E�      �  O�   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6��C-   
add_offset               min       ?�IQ���   max       ?��5�Xy>     �  Pl                              �         ]   x   _         !      ^                           	      
      '   |   >            /      �                  (   a            7      :   *   N��|N,%rN�ENi$N���O,�N`�O�X�N��P���O�NV\P�r�P�hP��N"jZO�O�Z:N��-PT�DOE�/N�nO5�`O�fO*\�OC�NT�O7�NS�mN���O|�OzO�OGPgsrP&7�N�G}N�6!N@��O��+Nd3�P/&	O��GO��JN���N���N���O���O\�5OD`�NYZNY�O�H:N���O���O0NWW<�y�#��w��t��e`B��`B��o�o%   <t�<#�
<49X<D��<T��<u<��
<�1<�9X<�j<ě�<���<���<�/<�/<�`B<�`B<�h<�<�<�=+=\)=�P=�w='�='�='�=8Q�=D��=H�9=H�9=Y�=q��=q��=y�#=�o=��=�\)=�hs=�hs=��=�-=�Q�=�Q�=��=�F>&�y����������������������������������������NDBN[dgtz~wtog[NNNNN-')/<=BGC<;/--------������������������������������������������������������������yqz|���������������y��������������������-*+0<N[t��������gB5-�����������������������������������������������BPN1)������������5@KMIA5������)>HLMW��tSJ,����������������������KHLOT[^dghktz~yth[OK�����	"/;CKG;/�������������������������������������������������������������������tst|��������������#0<ISOI<70$#�����
'/6::/+#
���!#/1<>DHH?</#!#0<AA?<910#�������
������������
/<HSU_]UH</'#
FGHU`ahfaUQHFFFFFFFF����������������������������������������YY[fht{��������uqh[Ysty��������������}ys���������������������#/2<CB9/����������	
��������������� ��������������������������������
/<HNUQIC<8/#����������������#=MPGH<�����5CNNSWRNB5)cabgov�����������tgc������� ���������"/272/"��
#$&%#
h[OD<63/4BO[qt{~|vqh����������

�����)6<??=8-)#��������������������##*+'#########����"#�������#//<CHMNH</#��������������������##/<HKLHHA<5/&#xnouz�����|zxxxxxxxx�
��#�����
��������������������
�
�n�{���}�{�n�l�h�i�n�n�n�n�n�n�n�n�n�n�������������������������������������������(�4�+�(���������������������ļ��������������������������������ûл׻ܻջлʻû������������������������/�6�;�H�T�X�T�H�;�/�"������"�$�/�/���#�(�5�N�P�W�V�N�A�-�(�$��������������������������������������������������6�O�h�}ă��x�[�?�6�1��������������M�N�Z�c�f�i�i�f�Z�U�M�L�A�<�7�6�7�A�D�M²º¿������������¿´²®«²²²²²²���	�"�;�a�m�l�g�\�H�"�����{�~��������������błŇŦŤœ�y�n�U�<�
�����������������
�#�0�0�����Ɓ�O������������*�OƎ�������������������������|������������������'�4�8�@�K�M�Y�f�r�Y�N�@�4�'������;�T�a�f�m�p�{�~�}�z�a�T�@�9�:�C�;�9�8�;ÇÏÓàìðìàÝÓÇÂ�z�s�zÂÇÇÇÇ���������Ľݽ���(�D�N�N�A�(���Ľ������A�N�Z�b�]�Z�R�E�A�5�(�������(�5�AF=F=FHFJFKFVFZF_FVFJFHF=F=F=F=F=F=F=F=F=�������������������ڼּּ���"�/�?�H�K�G�>�/� ��	��������������"�����������������������������������������r�����������������r�l�f�Y�W�V�Y�f�k�r���#�!����� ����
��������������
����������������������� �����'����� ������������f�r�������w�r�n�f�Y�M�J�M�Y�a�f�f�f�f�����������������������������������|�����"�.�5�G�I�O�T�V�T�G�;�.�*�"������"�g�s�����������t�g�Z�N�8�����(�A�Z�g������������ìÇ�n�R�G�A�<�5�7�H�UÇà�ž������˾׿!�-�"���׾����������~�������_�l�x���������x�l�_�S�J�S�\�_�_�_�_�_�_���	��"�(�(�"��	���������������������E�E�E�E�E�E�FFFFFE�E�E�E�E�E�E�E�E�`�m�y�������������y�m�`�T�I�G�@�@�G�T�`�5�A�N�N�R�R�N�A�=�5�.�*�5�5�5�5�5�5�5�5����%�*�)�!�	�����ֺɺƺ˺������ɺￒ�����������������������������x�v�z�������������Ŀѿ���ϿĿ������������������#�/�<�H�U�V�a�U�H�<�/�#� ��#�#�#�#�#�#��"�#�,�&�#��
�� �
���������ǔǡǭǶǳǭǤǡǔǈǅǄǈǎǔǔǔǔǔǔ�������!�)�-�:�F�L�N�M�F�D�:�-�!����D{D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D{DwDyD{�e�r�~�����������������~�r�Y�L�B�L�S�c�e�ܹ�����������ܹϹŹϹعܹܹܹܹܹ��a�]�b�i�n�{ŇŏŇ�{�n�a�a�a�a�a�a�a�a�a�)�5�N�h�b�a�H�5�)�����������������)¦¦­¦¡����4�M�P�M�@�=�4�(�������ݻܻ�����������
�
��
����������������������ؼ����!�)�,�!����������������������� ] b c [ @  z 4 L  D l G . � s I _ 8 = X X 9 U ( % [ M = U G : E % d u F X - > O 0 5 S  B .  1 d m N R U ( K    {     n  �  g  �  T  �  3  ^  o  �  �  
J  t  o  �  �  �  �  J  �  �  k  L  �  �  g  �  ?  R  �  �  �  �  �  �    |  R    �  �  �  �    �  �  �  \  !     �  D  ~�P�`��P��`B�t�;��
<D��%   <�<�j>_;d<�<�C�=��>
=q=�<ě�=<j=e`B=�P=��=��=�P=0 �=@�=8Q�=P�`=\)=q��=��=8Q�=49X=D��=��w>'�=��=T��=P�`=aG�=ě�=e`B>t�j=��T=� �=��=���=���=�;d>+=���=��
=���>z�=�S�>!��>$�/>=p�B�B;�B��B�]B*�nB"@,B�B��B6B	B}|B��BRSB�B�,B^�B�7A���B"FoB!8>B��B��B%�KB��B6�B%}B��B�|B�DB"9�B*E�BQ�B�B�B�0B�rB�B�^ByB"�B"�0BגB
��B�A���B)rB�Bu B6BB�B�IB��Bc�B�BXnBP�B=BC�B˦BX�B*ߥB"C�B��B�	B6�B	?RB��B 1�B�IB�cB@AB?�B��A�jB"?�B!@)B?ZBjB%@�B2�BBNB%��B��B��B�rB"8bB*"�BB?�B�B�BY�B'�B@KB?�B/	B"��B�GB
��B�(A���B>JB��B��BA!BC`BÉBAB�%B�'B?�B@�A�ZA��eA�9�A�b@�ΐ@��nA���A�V'A�>�A֟�A=A���A���A�1B�]A� �@�v�A�B�A�x�A.�jA�iKC���A��A��	A���@�?�A�^�A�K�A��@ߖcA!/�Ab"A��A���AQ�!@��)A\$YC���Aj��A�ba@Ki}Ar��Au��A�c�A�xB�:@p%*C��4?��?�A�A���A��|@�+OA��A
��A��A�..A�a�A��J@�W@��A�'~A���A�y�A�p�A=�xA�}�A�	A�pB� A��5@��<A��KAʁ	A.��A�~�C���A�A��,A��@��#A�r�A�x�A��o@ὛA 8�Aa�A��&AɘsAT�n@��.A]!C��Ak�A���@K�AAsAw��A��A�)B��@{��C��w@ �?��A��A���A�I�@���A�u=A
��                              �         ^   y   `         "      ^                           	      
      (   |   ?            0      �                  (   a            8      :   +                                 3         C   ?   W         !      4                                       #   1   1                  /                                 $      !                                    %         '   ;   S                                                      !      '                                                   $            N��|N,%rN�ENi$Nn��O,�N`�O�>ND�PD�O�NV\P��P� +P���N"jZN��sN޿�Nl	�O��OE�/N�nN�"O�;BO�xO g�NT�N��UNS�mN���O|�OzO�k�O�ԀO�N�G}N�6!N@��O$��Nd3�O]
O7�SO��JN���N���N���O���O!��OD`�NYZNY�O�H:N���O>)�O0NWW<  �  �  �      L  9  ~  �  i  �  �  R  �  g  �     B  �  v  :    *  �  &  �  w  [  f  L     ;    �  �  �  r  �  c  �    �      �  �  �  �  Q  ^      �  
�  
g  ��y�#��w��t��e`B�D����o�o:�o<T��=�9X<49X<D��=L��<�h<�j<�1<�`B=��<���=�hs<���<�/<��<�h<�=o<�=�w<�=+=\)=�P=#�
=�j=Y�='�=8Q�=D��=y�#=H�9>O�=�%=q��=y�#=�o=��=�\)=�E�=�hs=��=�-=�Q�=�Q�=�`B=�F>&�y����������������������������������������NDBN[dgtz~wtog[NNNNN-')/<=BGC<;/--------������������������������������������������������������������sz}��������������zs��������������������;89>DN[gt����~tgNB;����������������������������������������������)8>?5)�����������:FIC>5��������)<GKLRdu~tVL&����������������������MOPW[dhltwzvtih[QOMM	"/:;@;;/"	��������������������������������������������������������������tst|��������������##05<FD<:20#�����
#'/588/"
���#/<<BFG=</$##08<?@><70##�������
������������%%&/<HJUXURH><6/%%%%FGHU`ahfaUQHFFFFFFFF����������������������������������������YY[fht{��������uqh[Ysuz��������������~ys��������� �������������#/<?@>4/���������	
��������������� ��������������������������������
#$/<BHGC=</#�����������������

�������$)5=BHKI?5)cabgov�����������tgc������� ���������"/272/"��
#$&%#
h[OD<63/4BO[qt{~|vqh�����������������)6<??=8-)#��������������������##*+'#########����"#�������#//<CHMNH</#��������������������##/<HKLHHA<5/&#xnouz�����|zxxxxxxxx�
��#�����
��������������������
�
�n�{���}�{�n�l�h�i�n�n�n�n�n�n�n�n�n�n�������������������������������������������(�4�+�(�������������������������������������������������������ûл׻ܻջлʻû������������������������/�6�;�H�T�X�T�H�;�/�"������"�$�/�/�� �(�5�N�O�V�U�N�A�5�(���������������������������������������������������6�O�[�g�n�n�h�[�O�6��������������M�N�Z�c�f�i�i�f�Z�U�M�L�A�<�7�6�7�A�D�M²º¿������������¿´²®«²²²²²²�/�;�H�S�Y�Z�W�R�H�/�"������������	�"�/��0�UŇśŜŎŁ�n�U�<�����������������������-�-�����Ɓ�\�*�����������\Ɩ�������������������������|�����������������'�-�4�@�F�M�T�M�G�@�4�'�#����� �'�'�a�a�l�m�r�u�m�k�a�T�T�I�H�J�T�_�a�a�a�aÇÉÓàìðìàÜÓÇÄ�z�v�zÅÇÇÇÇ�нݽ������*�*�(�����ݽнǽ����˽��A�N�Z�b�]�Z�R�E�A�5�(�������(�5�AF=F=FHFJFKFVFZF_FVFJFHF=F=F=F=F=F=F=F=F=�����
��������������ܼ�����"�/�=�F�J�F�=�3�/���	������������"�����������������������������������������r�������������������r�o�f�Z�Y�Y�f�q�r���#�!����� ����
���������������
��������������������������������'����� ������������f�r�������w�r�n�f�Y�M�J�M�Y�a�f�f�f�f�����������������������������������|�����"�.�5�G�I�O�T�V�T�G�;�.�*�"������"�g�s�����������s�g�Z�N�:�(����(�A�Z�g�zÓàìð÷úúòàÇ�z�n�`�\�[�Z�a�n�z���������ξ��������׾��������������_�l�x���������x�l�_�S�J�S�\�_�_�_�_�_�_���	��"�(�(�"��	���������������������E�E�E�E�E�E�FFFFFE�E�E�E�E�E�E�E�E�`�m�y�z�����������y�m�`�T�R�L�K�T�V�`�`�5�A�N�N�R�R�N�A�=�5�.�*�5�5�5�5�5�5�5�5���������������ֺҺƺĺ˺ֺ�ￒ���������������������������~�}�����������������Ŀѿ���ϿĿ������������������#�/�<�H�U�V�a�U�H�<�/�#� ��#�#�#�#�#�#��"�#�,�&�#��
�� �
���������ǔǡǭǶǳǭǤǡǔǈǅǄǈǎǔǔǔǔǔǔ�������!�)�-�:�F�L�N�M�F�D�:�-�!����D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D}D�D�D��e�r�~�����������������~�r�Y�L�B�L�S�c�e�ܹ�����������ܹϹŹϹعܹܹܹܹܹ��a�]�b�i�n�{ŇŏŇ�{�n�a�a�a�a�a�a�a�a�a�)�5�N�h�b�a�H�5�)�����������������)¦¦­¦¡�������!�'�!�����������������������
�
��
����������������������ؼ����!�)�,�!����������������������� ] b c [ 8  z 4 J ! D l 4 ) � s 2 = ? 8 X X , U  # [ = = U G : @  ] u F X ( >  & 5 S  B .  1 d m N R 0 ( K    {     n  �  g  �  B  k  f  ^  o  �  :  	�  t  �    }    �  J    b  +    �  �  g  �  ?  R  �  �  z  �  �  �  `  |  �  �  �  �  �  �    X  �  �  \  !     �  D  ~  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  �  �  y  g  Y  `  _  J  5  !    �  �  �  �  �  q  R  7    �  �  �  �  �  �  �  �  �  �  �  |  t  l  e  ]  U  N  F  >  �  �  �  �  �  �  �  �  �  �  �  �  �  ~  l  Z  5  �  �  �          	    �  �  �  �  �  �  �  �  �  �  �  W  +  �  �  �  �  �  �    
    �  �  �  �  �  �  e  @    �  �  �  L  @  2  "    �  �  �  �  �  b  :    �  �  �  �    W  D  9  7  4  1  /  ,  )  &  $  !  !  #  $  &  (  *  ,  -  /  1  x  ~  u  c  R  =  .  1  *      �  �  �  �  J  �  o  �   �  X  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  b  H  K     l  H    �    R  i  Z     �  9  u  �  W  �    	�    �  �  ~  z  r  i  Z  E  -    �  �  �  t  O  '  �  �  �  �  �  �  �  �  �  �  �  �  �  �    +  H  X  _  f  n  o  o  p  p  J  d  �  �  �  �    H  O  6  
  �  �  k    �  �  �  �    {  �  �  �  �  9  �  /  �  
�  	�  �  �  !  G  x  �  P  S  L  Z  d  L  M  :    6  O  U    �  v  0  �  �  =  �  �  �    �  �  �  �  �  �  �  �  �  �  �  �  �  y  w  u  s  q  o  m  |  �  �              �  �  �  |  C    �  k    �  �  e    ;  G  I  t  �  �    :  @  ,    �  �  r    �  �  �  ]  �  �  �  �  �  �  �  �  m  J  "  �  �  �  �  ^  ?  "    �  I  �  �  �    7  U  k  r  u  r  k  U    �  p  �  �  I   �  :  &    �  �  �  �  �  _  9    �  �  �  c  0  �      �    ,  A  3  $      �  �  �  �  �  �  �  �  �  �  Z  �  �        #  (  *  (        �  �  �  �  �  w  L    �  �  �  �  �  �  �  �  �  t  S  0    �  �  �  �  d  4    &  �  �  "  &  $        �  �  �  �  �  p  W  <    �  �  "  �  �  �  �  �  �  �  �  �  j  D    �  �  X    �  ]  �  y   �  w  k  `  U  J  D  >  9  0  #    
  �  �  �  z  `  I  3    �  :  P  U  X  Y  Z  R  8    �  �  y  -  �  {    �  m  +  f  b  ]  Q  @  -    �  �  �  �  �  {  f  R  M  M  M  O  Q  L  C  :  0  (           �  �  �  �  �  �  �  �  �  �  �                   �  �  �  �  �  �  �  �  �  �  �  �  ;  *      �  �  �  �  q  X  E  7  +      �  �  �  M    �  �  �  �  �  �  b  3  �  �  y  ;  �  �  �  _    �  �  �  
H  
�    S  q  �  �  �  �  �  �  t  J    
�  
  	S  =      P  �  �  �  �  �  �  f  8     �  Y  �  g     �  7  �  �  �  �  �  �  �  �  �  �  v  g  [  U  X  a  l  x  �  �  �    P  r  j  b  Z  T  N  I  B  :  3  *           �  �        �  �  o  Z  P  E  4    	  �  �  �  S    �  �  g  '   �   �  ,  >  O  [  a  c  b  [  N  3    �  �  B  �  {  �  �  �    �  �  �  �  �  �  �  �  s  a  N  <  +      �  �  �  y  C  9  �  q  M  G  �  b  �      �  Z  �  >  Z    �  
�    �  X  r  �  �  �  �  |  k  T  9    �  �  �  �  k  ;  �  �   �      
  �  �  �  �  �  [  2    �  �  t  4  �  �  H    �    �  �  �  x  e  Q  ;  $    �  �    $    �  �  �  o  C  �  �  �  �  z  _  C  &    �  �  �  j  =    �  �  U    �  �  �  �  |  f  H  $  �  �  �  �  Z  0    �  �  Y    �  W  �  \    �  z  O  3  g  f  a  S  2  �  �  s  *  �  s  �  *    @  h  �  �  x  X  *  �  �  7  �    J  <    h  	�  �  �  Q  G  1    �  �  �  �  �  X  ,  �  �  r    �  Y  �  {  �  ^  Q  D  6  %       �  �  �  �  q  >    �  p    �  x  *    �  �  �  �  S    �  �  �  S  &  �  �  �  |  ^    �  �    �  �  �  �  a  R  e  �  �  �  e  5  �  �  �  =  s  �   �  �  �  �  �  �  l  V  A       �  �  �  �  �  c  B    �  :  	�  
V  
�  
�  
�  
�  
v  
c  
K  
&  	�  	�  	P  �  Y  �  �  3  �  �  
g  
O  
.  
  	�  	�  	�  	`  	%  �  �  .  �  �  2  g  �  �  �  �  �  v  L  *    �  �  �    �  d    �  p    �  u  '  �  �