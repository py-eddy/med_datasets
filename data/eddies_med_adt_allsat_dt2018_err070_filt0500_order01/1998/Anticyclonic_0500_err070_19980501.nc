CDF       
      obs    5   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type                     	amplitude                   comment       �Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM���   
add_offset               min       ?`bM���   max       ?�r� ě�      �  �   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       M��   max       P�^�      �  �   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       �#�
   max       =Ƨ�      �  T   effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @?�z�H   max       @F9�����     H   (   effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ?���R    max       @vG�
=p�     H  (p   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @1�        max       @P�           l  0�   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @��        max       @���          �  1$   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       �#�
   max       >�bN      �  1�   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A��   max       B15F      �  2�   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A��   max       B1?�      �  3�   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       ?BE�   max       C��      �  4t   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       ?O��   max       C�ٛ      �  5H   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max          �      �  6   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min             max          ?      �  6�   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min             max          3      �  7�   
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       M��   max       P:+�      �  8�   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6��C-   
add_offset               min       ?��-V   max       ?�/�{J#:      �  9l   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       ����   max       >hs      �  :@   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @?+��Q�   max       @F9�����     H  ;   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ?�\(��    max       @vDQ��     H  C\   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @'         max       @P�           l  K�   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @��        max       @�@          �  L   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         D�   max         D�      �  L�   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6��C-   
add_offset               min       ?�������   max       ?�/�{J#:     �  M�               x                     &                     "      Y                  %               3         	   
         8         e                  O         i   �   O�S1N��)N�X�NæP�^�O4�BOKhO��(OW	O@�OS-�O� (N3]Np�]O��|N�� OR��Ot�@PlN���P��IOE2^O�EdN�УN*%�N�l]PC��Oy�O[�NQ�jOO�OO��O=H_Nz��N��O �?N���M��O��Nc��O?<bP(�N�l�N�޹N��O�[OфO�nAN)�N$��O�X�O�)N$���#�
��`B�ě��D���o�o��o��o��o%   ;D��<t�<49X<e`B<�t�<���<��
<�9X<�9X<�9X<���<�/=o=o=o=+=C�=C�=��=#�
=0 �=49X=D��=H�9=L��=L��=L��=L��=P�`=P�`=P�`=T��=]/=]/=m�h=}�=�%=��=��=��=��
=�j=Ƨ�}}�����������������}��������������������bfgot������ytgbbbbbb����������������������������)87+
�����3/,*6BOSZ^_`[SOLB?63��������������������Z^aht�����������tc\Z������������������������������������������������������������"!'(%$/;HT\^`daQ;,"��������������������topqqt������uttttttt���������
	���������������������������ywz���������������zy
	)5BN][ZWNB5)acmqrv������������ga������������������������ 5HXWRPLB5������������������������rp����������������yr��������� ������
###
���������������������������&98&��������������	 ��������OO[]ht��������wth`[O��

�����������������������������	)5?DFFB5)��� (5BDNSWQRZNNB5)��������������������dcgikpst�������tkgddea`gtw�������|tsmge,./<HOPKH<3/,,,,,,,,B@COW\]\OCBBBBBBBBBB #/<BHQYdeaUH</$ 95337<EHJIHE?<999999MO[ht���������th_[OM������.68)��������������������������5)5:BNPY[\[NB=:55555���������������������������� ���������������������������������)23,$������������������������������������������������ ��������������
!"
������
	
�hāćďĔĚĤķĳĤĚĖĈ�t�h�B�F�O�[�h�������������������������������U�b�c�n�s�p�n�b�U�I�G�I�J�Q�U�U�U�U�U�U��*�2�6�:�C�C�C�6�/�*�'�����������������B�[�b�5��������d�Z�U�^�z�����������ʾӾԾоʾ�����������|����������(�5�A�N�T�Z�`�`�]�Z�N�A�=�5�%�����(���(�6�;�G�A�4�(������ݽؽ߽�����Z�f�s�����������������z�f�c�Z�U�P�O�Z�������ûŻлۻڻлû���������������������������
�������������ŹůŭŬŹż�����	��"�;�H�T�a�j�h�[�L�;�/�	�����������	�������������������������������������������	��"�/�8�8�/�"��	�	�������������������������	��� ��	�������������������׼�'�4�@�E�M�T�M�L�@�4�'���������6�O�\�h�o�q�j�e�\�O�J�C�<�6�0�*�'�*�4�6�.�;�G�P�T�Z�\�]�_�T�P�G�;�/�)�"���#�.�m�y�����Ŀѿ޿޿ӿ������y�m�`�X�U�M�[�mÇÓàèìõìãàÓÇ�z�y�u�z�{ÇÇÇÇ�����$�8�$�������ƳƚƎƁƀƉƅƉƚ�������ʼּۼݼټڼּʼ������������������4�A�Z�f�s����������s�f�4��� �	��(�4���������������������s�f�Z�R�Z�f�s�v��(�4�A�F�A�9�4�(�(�$� �"�(�(�(�(�(�(�(�(��������������¿¶²¬¦¡¦­²¿�������g���������d�]�Z�N�A�5�������6�N�g����������� ���������������������������Ҽ��������¼ü������������������������������ ������������������������������������	����� ���	�������������������
��I�R�V�T�K�<�0�#�����
�
����������	�� �"�+�"��	��������������������M�O�Z�f�l�f�a�Z�M�G�A�8�A�L�M�M�M�M�M�M�Z�g�q�s�����������������s�g�e�^�Z�Y�Z�Z�����������	��	������������������������������� ��������������������������������`�m�s�x�m�a�`�`�Z�_�`�`�`�`�`�`�`�`�`�`��(�5�A�O�l�s�g�`�N�A�(���������������������������������������������������<�H�O�P�L�Z�^�U�<�/�*�&�&�)�/�+�/�4�9�<�~�������������������~�b�W�P�8�.�.�@�e�~���������!�$�!� �������ٺ������ ������������������������
��#�)�#�"�"��
���������	�
�
�
�
�
�
�)�6�B�O�[�Z�O�M�J�B�:�6�)�#���� �)�)�y�����������������������������|�y�n�n�y�����ûܻ�������ܻл��������x�g�_�x���:�F�S�_�Y�S�F�:�7�4�:�:�:�:�:�:�:�:�:�:ĚĦĳĳĳĳĦĚĕĚĚĚĚĚĚĚĚĚĚĚ�������ʼ��������ּʼ���������������D�D�D�D�D�D�D�D�D�D�D�D�D�DtDpDtD{D�D�D��I�H�H�I�U�b�g�n�o�n�b�U�I�I�I�I�I�I�I�I l 6 1 r R B & . 2 0 T Q � n 6 A E M < # , I < Z U J r 8 : D ] T C 5 a j * L 8 i u B C f @ ) - S O 6 H E @    �  �  �  ~  7  �  �  �  �  �  �  �  x  �      �    �  �  W  �  g  =  `  �  (  Q  E  }  �  �  �  r  !  }  �  '  r  �        �  �  :  7    <  =  �    N�#�
:�o��o;o=��<�t�<�/<��
<D��<�C�<���=<j<T��<�o='�=o=t�=#�
=e`B='�=�l�=#�
=ix�=�w=t�=D��=�\)=8Q�=��=<j=ix�=\=m�h=e`B=m�h=q��=�%=]/=��=}�=�\)>�=�\)=y�#=��=��=���>t�=�\)=���>=p�>�bN=�
=B8B�AB	�pB5�BYBsTB�jB�B �B!��B�'A��B��BfB��B!ܱB ��BoB
�=B"0�B�+B"s�B��B��B�B	�B�By�B3B$@fB�lBe�B|hBڙB	��B
B�B15FB��B�3BU�Bx�B��B�B]B#�B*��B��Bj�B �QB��BhB�B<B��B	�\B
��B@DB@B�BE�B �B!�zB��A��B�DBCFBO�B!��B	B��B
�B"@B��B"=mBc�B�<B�RB(B*+B��BJ�B$��B��BASBYjB�jB	ՉB	�B�#B1?�B�dB��Bg�BTOB @2B.�B�.B?#B*��B@)BE6B �
B�
B@B;]A�aG?BE�A�FA��CA�t}AK�JA�*�A1�yADw�@�$vA��$A��ZAs�WA��]A�.@ʢB>�Ac�AAq}�A�pB�@��A<$�AD��A8_-A���A���Aл�@�8�A)hA���A��kA���A=�A�& A��QAъ[Aj/A�9�A�l�A�p�?��(@W��Aӽ�A��A���ApB@���@��OA�a�@�;C��A���A�t�?O��A�z<A�t�A��AL��A�x~A1�AC��@���A�v�A�IfAt�IA��A�v�@��BdAcEAsAʬBP@��A:�KAE�A8�A�S,A��9A�h@�iA�A���A�A��vA>5eA���A�4kA�Q5Ai.aA�kfA���AÃ#?�Y-@U�-A��A�wAׁ�A	�@�a@��UA�b'@���C�ٛA��7      	         y                     &                     #      Y                  %               4         	   
         8         f                  P         j   �   	   #            ?         #            !                     +      3      '            3               #                     !         /                  %            !                                                            +      #      '            3                                                                              N��jNY�N�X�NæO\dO�OF�O���O�O@�O��O�(N3]Np�]O!�?N���N߿�Ot�@PlN���O���O.5�O�EdN�УN*%�N�l]P:+�Oy�N��dNQ�jOO�OOs��O=H_Nz��N��O �?N\�$M��O�<�Nc��N�MO���N�l�N�޹N~+mN�~OфO��N)�N$��O:U�O�IN$��  a  A  �  �  	x  E  �  �  �  -  �  #  ?  "  B  �  �  [  "      #  �  �  �  q  �  ]    {  R  Y  b    X  �  �  �  	  3  �  
s  d  �     ?  M  L  �  %  �  �  `���ͻě��ě��D��=�7L%   ;ě�;o;o%   ;�`B<���<49X<e`B<ě�<�9X<���<�9X<�9X<���=y�#<�`B=o=o=o=+=\)=C�='�=#�
=0 �=L��=D��=H�9=L��=L��=T��=L��=m�h=P�`=m�h=�E�=]/=]/=q��=�C�=�%=��
=��=��=�`B>hs=Ƨ�����������������������������������������bfgot������ytgbbbbbb���������������������������

����41/5BOQX[\]^[YOOB@:4��������������������b`hkt������������thb������������������������������������������������������������-/3:;HKTUWWVTMH;:0/-��������������������topqqt������uttttttt��������� ������������������������������������������������
	)5BN][ZWNB5)acmqrv������������ga������������������������)1@C@9)������������������������rp����������������yr��������� ������
###
����������������������������#46/��������������	 ��������TU[aht������thc[TTTT��

����������������������������)25;ADDB5� (5BDNSWQRZNNB5)��������������������dcgikpst�������tkgddea`gtw�������|tsmge-/0<HLNJH<6/--------B@COW\]\OCBBBBBBBBBB! "%/<HOV\`a\UH</*#!95337<EHJIHE?<999999{�����������{{{{{{{{��������
	���������������������������5)5:BNPY[\[NB=:55555������������������������������������������������������������������� ,.,)����������������������������������������������������������������

����
	
�h�tāăćĉā�t�h�b�[�Z�[�]�h�h�h�h�h�h�������������������������������U�b�c�n�s�p�n�b�U�I�G�I�J�Q�U�U�U�U�U�U��*�2�6�:�C�C�C�6�/�*�'�������������������������������������������������������ʾѾ;ʾ������������������������(�5�A�N�N�Z�[�\�Z�W�N�A�5�.�(� ��$�(�(���$�(�1�4�6�(�&�������ݽ߽������f�s�����������������s�k�f�\�Z�Y�Z�^�f�������ûŻлۻڻлû�������������������������������������������ŹűŲŹ���������"�/�;�H�K�K�H�=�;�/�"���	��	�	���"�������������������������������������������	��"�/�8�8�/�"��	�	������������������������������������������������������˼'�4�@�A�G�@�@�4�'�������#�'�'�'�'�6�C�O�\�_�h�j�j�h�^�\�O�I�C�C�;�6�6�6�6�.�;�G�P�T�Z�\�]�_�T�P�G�;�/�)�"���#�.�m�y�����Ŀѿ޿޿ӿ������y�m�`�X�U�M�[�mÇÓàãìòìààÓÇÁ�z�x�z�ÇÇÇÇ���������������ƳƧơơƤƩƳ�����弱���ʼּټܼؼռʼ����������������������4�A�Z�f�s����������s�f�4��� �	��(�4���������������������s�f�Z�R�Z�f�s�v��(�4�A�F�A�9�4�(�(�$� �"�(�(�(�(�(�(�(�(��������������¿¶²¬¦¡¦­²¿�������g�����������z�d�N�A�5�(�����7�I�Z�g����������� ���������������������������Ҽ������������������������������������������ ������������������������������������	����� ���	������������������#�0�<�I�M�R�P�I�F�<�0�#���
���������	�� �"�+�"��	��������������������M�O�Z�f�l�f�a�Z�M�G�A�8�A�L�M�M�M�M�M�M�Z�g�q�s�����������������s�g�e�^�Z�Y�Z�Z�����������	��	����������������������������������������������������������������`�m�s�x�m�a�`�`�Z�_�`�`�`�`�`�`�`�`�`�`�(�5�A�N�c�i�Z�N�A�5�(������	���(�����������������������������������������<�E�H�Q�N�H�<�/�+�+�/�5�<�<�<�<�<�<�<�<�e�r�~�������������~�r�e�Y�N�F�C�H�L�U�e���������!�$�!� �������ٺ������ �������������������������� ���
���������
����������)�6�B�O�R�S�O�B�;�6�)�(�!�'�)�)�)�)�)�)�y�����������������������������|�y�n�n�y�������ûлܻ����ܻлû��������������:�F�S�_�Y�S�F�:�7�4�:�:�:�:�:�:�:�:�:�:ĚĦĳĳĳĳĦĚĕĚĚĚĚĚĚĚĚĚĚĚ���üʼּܼ����ּʼ�����������������D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D{D�D�D��I�H�H�I�U�b�g�n�o�n�b�U�I�I�I�I�I�I�I�I % 3 1 r " ? # / , 0 H 3 � n 0 I ; M < " # J < Z U J q 8 8 D ] * C 5 a j # L 1 i I  C f 9 3 - ; O 6 6 7 @    �  s  �  ~  �  P  3  "  Q  �  G  A  x  �  U  �      �  �  $  �  g  =  `  �  J  Q    }  �  �  �  r  !  }  k  '    �  �  3    �  �  �  7  3  <  =  �    N  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  �  �  �    1  Z  N  9  $  ,  O  )  �  �  �  J  �  �    4  /  6  >  ?  <  7  .  $      �  �  �  �  v  I    �  �  �  �  �  �  �  �  �  �  �  �  w  m  c  W  J  =  0      �  �  �  �  �  �  �  �  �  �  �  �  s  d  N  4    �  �  �  �  �  �  �  |     k  �  ?  �  	  	I  	j  	w  	b  	  �  8  �  n      �  !  8  E  E  A  8  +      �  �  �  �  t  K    �  �    e  }  �  �  �  �  �  �  �  �  j  A    �  �  :  �  �  )  �  �  �  �  �  �  �  �  �  �  �  �  j  I  *    �  �  �  b  8    �  �  �  �  �  �  �  �  �  �  �    d  D    �  �  }  B    -      �  �  �  �  �  j  1  �  �  �  �  �  �  �  {  =  �  �  �  �  �  �  �  �  �  �  �  �  {  R    �  {    �  C  �  �  �  �  �          #      �  �  y  -  �  Y  �  1  i  ?  ?  >  >  =  =  =  <  <  ;  2  !     �   �   �   �   �   �   �  "          
        �   �   �   �   �      	          �    
    6  B  ?  8  *    �  �  �  w  0  �  �  6  �  M  �  �  �  �  �  �  �  �  �  t  S  '  �  �  �  P    �  �  -  �  �  �  �  �  �  �  �  �  �  �  �  ~  [  ,  �  �  �  y  f  [  W  O  G  =  2  )  "        �  �  �  W    �  `  �    "       �  �  �  �  �  �  m  @    �  �  \  6  �  k  �   }  �  
            �  �  �  �  O    �  Y  �  �  <  �  �    h  �  �  �  �  �        �  �  ^    �    }  �  �  �           �  �  �  �  �  �  �  l  Y  G  (    �  �  ;   �  �  �  �  �  �  �  |  l  Z  F  /    �  �  �  �  �  K  �  �  �  �  �  �  |  m  _  S  F  >  7  /  &      �  �  �  �  q  �  �  �  �  �  �  �  y  o  d  X  L  @  3  '        �  �  q  d  T  ?  &  	  �  �  �  �  r  d  P  7      �  �  �  Y  �  �  �  �  �  }  m  Q  *  �  �  �  (  �    �  �  K  �     ]  ]  [  V  O  G  =  2  %      �  �  �  �  �  �  u  L                �  �  �  ]  &  �  �  f    �  �  U  �  �  {  x  u  q  i  a  X  N  D  :  .  !      �  �  �  �  �  �  R  G  8  &    �  �  �  �  �  o  U  /  �  �  �  P  �  �    �  E  V  L  5    �  �  �  �  h  &  �  p  �  r  �  _     �  b  [  T  I  <  %      �  �  �  �  �  �  �  �  �  p  /  �    �  �  �  �  �  �  �  �  �  }  m  X  D  +  
  �  �  �  S  X  H  8  &    �  �  �  �  �  �  �  {  j  Y  ;    �  v    �  �  x  u  s  e  S  A  /         �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  z  b  F  *    �  �  �  �  �  �  �  D  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  	  	  �  �  �  |  K    �  �  a    �    p  �  K  1  3    �  �  �  �  V  $  �  �  �  G    �  �  W    �  �  |    �  �  (  �  �  �  �  �  �  �  �  o  H    �  �  :  �  �  �  �  	m  	�  
$  
X  
m  
r  
l  
S  
$  	�  	�  	@  �  E  �  �  �  �  d  <      �  �  �  �  v  [  ?  !    �  �  �  �  �  	  $  �  �  �  �  �  �  �    f  L  3    �  �  �  �  �  �  �  �            	  �  �  �  �  �  �  �  �  t  l  c  �  �  �        !  ,  <  :  '    �  �  �  �  �  �  �  s  �  �  �  M  G  B  ;  4  .  '    2  1  "    �  �  �  |  >  �  �  q  
�      6  I  9  !  
�  
�  
�  
?  	�  	u  �  s  �  '  f  �  G  �  �  �  �  �  �  �  �  �  �  r  ^  .  �  �  j  8    �  �  %                �  �  �  �           �  �  �  �    �  =  �  �  �  �  �  �  �  ?  �  I  �  �  �  
d  �  �  '  �  �    h  �  �  �  �  z  O  �  D  �  �  �  ^  5  r  =  �  `  U  K  ?  1  #      �  �  �  �  �  �  {  N    �  �  �