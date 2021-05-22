CDF       
      obs    8   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type                     	amplitude                   comment       �Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM���   
add_offset               min       ?h�t�j~�   max       ?Ցhr� �      �  �   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       M�9   max       P�=/      �  �   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       ���   max       =�      �  l   effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @?0��
=q   max       @F@          �   L   effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ?�33334    max       @vO�z�H     �  )   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @"         max       @N�           p  1�   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @�E        max       @�`          �  2<   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       ��C�   max       >�      �  3   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A�x�   max       B0�T      �  3�   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A�	�   max       B0�X      �  4�   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       >��   max       C�Z�      �  5�   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       ?pI   max       C�X      �  6�   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max         Q      �  7|   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min             max          ?      �  8\   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min             max          7      �  9<   
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       M�9   max       PiX      �  :   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6��C-   
add_offset               min       ?��vȴ9X   max       ?�S����      �  :�   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       ���   max       >7K�      �  ;�   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @?0��
=q   max       @F@          �  <�   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ?�33334    max       @vO�z�H     �  E|   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @         max       @N            p  N<   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @�E        max       @�7           �  N�   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         F�   max         F�      �  O�   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6��C-   
add_offset               min       ?�!-w1��   max       ?�S&���     �  Pl         &   =         6   J                        ;        Q                  "   1   C         ;      $   +               !          ,                        !      ?   
   �      
   O��N�XqO���O�G�NF Ou�P%$P^��P+W�NH7NCu�O�W:O�N��N;Z�P5�qO���O�9P�\O�i�OEN4ҒN�@N�,�OĢO��mP�N$��N�p�P�=/N��]Ok��O�U�O�e�M�9N��Oe�O�Q�OO�aO�+aO���Nb�O
�~Nf�N�M^OI��N�޺N��O���N�X�O���N�v|O��VO�PN���NLڽ�㻃o%   ;D��<o<o<t�<t�<t�<D��<D��<D��<e`B<e`B<u<�C�<�C�<�t�<���<ě�<��=o=o=o=o=+=\)=�P=�w=0 �=0 �=49X=49X=49X=49X=8Q�=@�=D��=H�9=L��=L��=P�`=P�`=T��=Y�=]/=]/=aG�=�o=��=�O�=�O�=��
=ȴ9=ȴ9=�#/<=HIPRH<6/.#��������������������#/<HPU_jaUH</#����
#/<DHD</#
������� ������������
#0<ILIC=<0#
�����,8B=)������ZX\et������������xeZ~~v����������������~LIKUYajkaULLLLLLLLLLgginvz����zngggggggg������������������������
 #'*#��#(/021/$#"�������������������������
$(.)
�����#/;HNRSSNH</��������������������,).B[t����������gN6,����
#)65#
��������~}}{��������������;ABOO[`][ODB;;;;;;;;b_chmtw�������thbbbb�����������������������������������������������
&))#
���hkicbnz�����������nh46CIOV\b\OCB96444444}�����������������}}����):QWWMF5)������������������������������������������7=@;5+���������������������������������������������:77<BNOOSVSONB::::::[VWWX\amz|������zma[����+=BEJKD;6)��zz|���������������]]ct{������������tg]��������������������"&)5BIGB5)""""""""""��������������������!!������������������������������')6BCKHFB<63+)''''''��������������������6/4ETam������zmaTH;6��������������������)4@BNM6)��RQ[ght}}wtmha[RRRRRR���������

��������� ���)+55855)HHB<4/../6<AEHHHHHHHD�EEEE"E"EEEEED�D�D�D�D�D�D�D�D��_�l�x�}�}�z�x�l�_�V�W�S�_�_�_�_�_�_�_�_��������������	����������������޿`�m�y�����������������y�`�T�H�>�9�=�G�`���������û������������������������������x�����������~�o�l�h�_�S�Q�M�L�S�a�l�t�x�/�H�O�Æ�z�j�a�U�<�/�'�%������#�/��/�T�a�c�]�P�H�;�0�������������������"�.�T�j�j�v�r�T�;��	��׾ʾ����ƾ���"���)�/�)�#������������������������������������������������������a�m�z�������������������z�m�a�T�L�O�T�a�;�H�T�V�a�j�j�a�T�H�;�8�/�"� �"�$�/�2�;�A�E�N�P�N�F�A�5�(�"����(�5�7�A�A�A�A�����������������������������������������B�[�b�t�o�p�g�[�N�)���������������)�B�����������������������������������{�����������!�#�!������������������6�O�^�c�[�O�B�$��������������������Y�d�_�W�M�H�J�@�4�'���� ���+�4�@�D�Y�������������������������������y�q�s�~����'�'�'�'����������������'�3�@�F�L�X�Y�[�Y�L�@�3�'�%�$��'�'�'�'�6�A�I�B�A�<�4�(�����������(�4�6���Ľнݽ����"������ݽĽ��������������ʾ׾����������㾱���������������������ֺ��������غֺܺкʺ��������y�|�z�y�x�m�b�`�_�]�`�`�m�u�y�y�y�y�y�y����������������������������������������Ƴ������$�!�����Ƴƚ�u�h�6���CƁƚƳ�O�[�h�s�t�}āćā�t�h�\�[�O�L�K�O�O�O�O�/�;�H�T�a�m�s�y�z��z�m�a�T�H�@�1�&�#�/���������������������������������������ѿ��ݿ���(�6�;�5�.�(������ѿĿ�������²¿��������¿²®²²²²²²²²²²²�ܻ����	��������ݻܻԻܻܻܻܻܻ�ŇŔŠŭŹ����������źŭŠŔňŇŀ�}ŁŇ�r�~�����������ɺԺɺ������~�r�g�_�c�k�r��������������������������������Ҿ�(�4�F�M�Z�_�]�P�A�4�(���������l�y�������������������y�`�S�G�G�S�`�k�l����#�#���������������ÇÓàìñùûùóìàÓÇ�|�z�y�z�|ÀÇ������������ݿܿܿݿ��������������a�n�r�u�s�z�|�z�n�e�a�]�W�V�V�^�a�a�a�a�;�G�T�[�`�m�f�`�S�G�;�.�"��
���"�.�;������������������������ܹ޹���������ܹϹϹù��ùϹҹܹܹܹ�čĚĦĳļ��������ĿĳĦĚĎĂ��~āĈč�#�/�<�D�H�U�Z�a�b�a�U�H�<�/�#�!��#�#�#�r���������������������r�f�M�@�>�C�R�f�r�f�r�t������r�f�`�Y�R�Y�^�f�f�f�f�f�fDoD{D�D�D�D�D�D�D�D�D�D�D�D�D{DrDgDdDjDo��#�0�1�;�;�5�0�#����
�	�������������������������������������������������������
���!�*�/�*�������� 1 - 0  R N > + ^ C F % I ; c /  -  ] T M A N D ] X s E e I F $ z 8 E & J / 9 & G $ 7 ] L N G Q F - Q   $ :  /  �    m  o  �  �  �  Y  [  h  T  M  �  �  (  @  L    3  �  h  �    �  �    �  �    �  �    �    �  �  �  �  �  R  �  .    �  �  �  �  �    �  �  v  I  �  V��C�;o=�P=�%<T��<e`B=}�=���=+<�C�<u=+<���<��
<��
=���=#�
=+>�=,1=8Q�=��=H�9=�P=��=��T=��=�w=L��=��`=e`B=��
=�-=�O�=P�`=}�=��=��
=�t�=��T=�v�=]/=��w=e`B=�o=��w=�+=}�=ě�=���>+=���>}�=�x�=�"�>�Bi�B�MB��B�NB#j�B%s�B�B
�3B ��B��B
"Bg�BB�B�B��BB{B ��B	;B$��B[�B�BGPB��B"mnB�3B{�B0�TB��BB�SB�B�}B�(B eB^�A�̂B�B �QB
��B,\sB6SB!�mB�B*�B��B�fB݋A�x�B�rB�5B�B��B"�B�B��B?�B�pB��B�B#DmB%��B�%B
��B � B�wBƀB~�B��B�B��B�BB�B �GB	=�B$��B?�B�B�B��B"�wBH�BARB0�XB�iBJ�B7#B?�B�}BI�B��B?
A�k;B�"B �8BM�B,J�B-8B!ƋB�Bn�B��BΤB��A�	�B��B�PB<B��BŜB��B�0C�Z�@��AҮ�Akr(@��@��6A���A�Z�A]��A�WkA�r�A���A�|A��A�"pA�p A�L�@U��A�KR@�05A���?��@?� A7��A,1*AQ��@A�ZAk@l@�B�SA�f�A�67A�5�A�F�A�t�@�7$A���@-�A�'EA8)	A�)A3��A���A�"A��AcR@Z�S>��A�݅Aé@㇆@�ҙC��sA�#�A�A���C�X@���A҉�Ak��@��v@��Aũ�A�{�AZ��A�q;A�y4A�y>A�r*A���A�:�A��A�V@[(A�v�@�T4A�k�?wN�?�R�A7>kA,y�AR�@C��Ak 9@�kB��A�vMA��A�.A���A�w�@�2fA��@O%A�N/A8�=A��A3�A��AW�A�xAa��@\�?pIA��hA�@��@��C��?A�y�A�g�A���         &   =         6   K                        ;        Q                  "   1   C         <      %   ,               !          ,                        !      @      �      
                        +   3   /                     )         1                  !   !   -         ?         !   )                     !                              !                                    #   !   /                     !                                 !         7            '                                                                  O��N�XqO v�O[eRNF Ou�Oχ�O�+)P!h�NH7NCu�N���O�N��N;Z�O�o_OGP�O�9O��}O	�'N��N4ҒN���N�/�OK�"Op��O�زN$��N�p�PiXN��]O5uO�6�O�vxM�9Nq	oOW77O�Q�OG�O�>TOZ%0Nb�O
�~Nf�N�=�OI��N�޺N��O���N�X�O���N�v|O^NO�N���NL�  Z  D  ?  	�  �  h  �  <    ?  �  ;  �  �  �  �  �  �  z  L  M  X  �  �  �  �  �  6  8  �  �  �  �  %  �  �     �  �  A  �    y  �  -  �  �  �  l  �  7  �  {  �  �  ��㻃o<T��<��
<o<o<���=�w<#�
<D��<D��<ě�<e`B<e`B<u=o<�j<�t�>7K�<�=C�=o=C�=+=0 �=0 �=L��=�P=�w=L��=0 �=H�9=H�9=8Q�=49X=@�=D��=D��=]/=]/=q��=P�`=P�`=T��=aG�=]/=]/=aG�=�o=��=���=�O�>%=���=ȴ9=�#/<=HIPRH<6/.#�������������������� #'/<CHMUVURH><3/#$   
#/;<@@=</#
������ ������������
#0<ILIC=<0#
������)064"����fehmt������������tjf�x����������������LIKUYajkaULLLLLLLLLLgginvz����zngggggggg������������������������
 #'*#��#(/021/$#"�������������������������

�������#/<DHKLLHG</#��������������������@?AHN[gt������tg[NC@����
#,0(# 
�����������������������;ABOO[`][ODB;;;;;;;;e`dhot}�������|tohee������������������������������������������������

���klnz�������������ztk46CIOV\b\OCB96444444}�����������������}}����)4BLSSI5)��������������������������������������������"19;7)�������������������������������������������<89=BOQUROLB<<<<<<<<\VXXZ^amz{������zma\����+=BEJKD;6)��~�����������������ejt��������������tie��������������������"&)5BIGB5)""""""""""��������������������!!������������������������������')6BCKHFB<63+)''''''��������������������6/4ETam������zmaTH;6��������������������)6;>=6)���RQ[ght}}wtmha[RRRRRR��������	
������������������)+55855)HHB<4/../6<AEHHHHHHHD�EEEE"E"EEEEED�D�D�D�D�D�D�D�D��_�l�x�}�}�z�x�l�_�V�W�S�_�_�_�_�_�_�_�_�������
�������������������������`�m�y�����������������y�m�`�_�R�J�J�S�`���������û������������������������������x�����������~�o�l�h�_�S�Q�M�L�S�a�l�t�x�<�H�a�h�{Á�}�s�c�[�U�H�<�/�&�$�!�*�/�<��"�/�;�H�L�K�E�;�/�"�	�������������	��"�.�T�g�h�s�m�T�;��	�׾ʾ����ɾ׾���"���)�/�)�#������������������������������������������������������m�z�����������������{�z�p�m�l�m�m�m�m�m�;�H�T�V�a�j�j�a�T�H�;�8�/�"� �"�$�/�2�;�A�E�N�P�N�F�A�5�(�"����(�5�7�A�A�A�A�����������������������������������������C�N�[�d�f�`�N�B�5�)��
���������)�5�C�����������������������������������������������!�#�!������������������)�1�;�@�>�4�)������������������'�4�@�K�F�@�?�<�4�'���������'�'���������������������������~�v������������'�'�'�'����������������'�3�@�E�L�V�Y�Z�Y�L�@�3�'�&�'�'�'�$�'�'��(�4�A�G�A�@�7�4�(�����������нݽ�������������н˽Ľ������ýо��ʾ׾������������ʾ������������������������������ۺϺ���������y�|�z�y�x�m�b�`�_�]�`�`�m�u�y�y�y�y�y�y��������������������������������������������������������Ƴƚ�u�D�,�3�CƁƚ���O�[�h�s�t�}āćā�t�h�\�[�O�L�K�O�O�O�O�;�H�T�a�d�m�q�u�u�m�a�T�H�C�4�/�+�/�/�;���������������������������������������������ݿ���(�4�:�5�,�������ѿſ�����²¿��������¿²®²²²²²²²²²²²�ܻ�����������ܻջܻܻܻܻܻܻܻ�ŇŔŠŭŹ����������źŭŠŔŉŇŁ�~łŇ�r�~�����������ɺԺɺ������~�r�g�_�c�k�r��������������������������������޾(�4�A�M�V�Z�X�K�A�4�(���������(�y���������������������y�j�`�Z�T�`�l�t�y����#�#���������������ÇÓàìñùûùóìàÓÇ�|�z�y�z�|ÀÇ������������ݿܿܿݿ��������������n�p�s�r�z�n�d�a�_�Y�X�Z�a�b�n�n�n�n�n�n�;�G�T�[�`�m�f�`�S�G�;�.�"��
���"�.�;������������������������ܹ޹���������ܹϹϹù��ùϹҹܹܹܹ�čĚĦĳļ��������ĿĳĦĚĎĂ��~āĈč�#�/�<�D�H�U�Z�a�b�a�U�H�<�/�#�!��#�#�#�������������������r�f�Y�K�G�L�Y�f�r��f�r�t������r�f�`�Y�R�Y�^�f�f�f�f�f�fD�D�D�D�D�D�D�D�D�D�D�D�D�D{DpDnDpD{D~D���#�/�0�:�:�4�0�#���
����
���������������������������������������������������
���!�*�/�*�������� 1 - &  R N 4 ) a C F 5 I ; c &  -  D G M D = A S K s E b I 9  x 8 < # J - 1  G $ 7 i L N G Q F % Q   $ :  /  �    �  o  �  �  �  2  [  h  �  M  �  �    �  L  s  +  �  h  �  �  �      �  �  x  �  �  �  f    �  �  �  O  B  �  �  .    �  �  �  �  �      �  �  &  �  V  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  Z  S  J  =  *    �  �  �  �  {  Q     �  �  '  �  Y  �  r  D  B  @  >  :  6  2  +  #      �  �  �  �  m  A     �   �  �  @  �  �  �    8  <  '    �  �  o  ,  �  q  �  2  K  D  �  �  	  	N  	y  	�  	�  	�  	t  	O  	  �  s    t  �  -  a  q  �  �  �  �  �  u  _  I  3    �  �  �  �  t  b  P  @  1  "    h  _  V  L  C  9  0    	  �  �  �  �  �  �  �  }  i  V  C    �  �  �  �  �  �  �  �  �  �  �  x  G     �  �  "  @  �  Q  �  �  �      /  :  ;  &  �  �    +  �  M  �    =  =  	      �  �  �  �  �  �  �  �  �  i  9    �  �  k  ,   �  ?  <  :  8  5  1  .  *  &            �  �  �  �  e  D  $  �  �  �  �  �  �  �  �  �  �  �  �  �  �  w  m  d  Z  P  G  �  �  �  �  �  �  �      &  3  :  3    �  �  �  U  �  \  �  �  �  �  �  �  �  v  l  j  h  f  b  \  W  R  N  K  H  E  �  �  �  �  �  z  l  Y  C  ,      �  �  �  �  �  �  j  H  �  �  �  �  �  �  �  �  �  w  g  V  ?  $  	  �  �  �  S    �  �  E  �  �  �  �  �  |  R    �  �  _    �  ;  �  �  K  f  z  �  �  �  �  �  �  �  {  h  K  +    �  �  r    �   �  �  �  �  �  �  �  �  �  z  e  K  /    �  �  �  �  �  X       x  D  �      �  �  ,  y  ]  �  g  �  B  �  ;  R  �  �  %       +  4  F  L  D  5  $    �  �  �  �  �  c  <    6  F  >  :  =  L  I  ?  1       �  �  �  �  �  �  r  O  '  �  X  I  :  +      �  �  �  �  �  {  e  Q  E  8  .  -  -  -  �  �  �  �  �  �  �  o  V  7    �  �  �  �  �  v  j  n  �  �  �  �  �  �  �  �  �  �    u  j  a  X  N  E  A  =  :  7  >  s  �  �  �  �  �  �  �  �  �  b  %  �  �  W    �  �  E  ?  �  �  �  �  �  �  �  �  p  1  �  �    �  9  �  "  h  �    P  �  �  �  �  �  �  �  p  3  �  �  &  �  
  V  �  �  W  6  5  3  2  1  /  .  ,  +  )  "       �   �   �   �   �   �   �  8    �  �  �  �  o  J  ,    �  �  �  �  �  n  R  6  o  �  r  �  �  �  �  t  R  %  �  �  �  _  :    �  �  &  �  �   �  �  �  k  >    �  �  �  �  v  S  #  �  �  �  ]    5  '  �  �  �  �  �  �  �  �  �  |  >  �  �  C  �  �    �  ]    �  �  �  �  �  ~  q  f  `  b  ^  R  :    �  �  X  �  P    �    "    �  �  �  �  �  w  L    �  �  �  ~  <  �  �  y  {  �  �  �  �  �  y  o  d  Y  M  A  4  #      �  �  �  �  �  g  �  �  �  �  �  b  >    �  �  \    �  ?  �  >  �  '   �  �  �  �  �  �  �  �  p  P  (  �  �  �  e  &  �  �  @  �  �  �  �  �  �  |  h  S  B  2    �  �  �  I    �  �  C    �  �  �  �  �  �  �  �  �  �  �  |  W  %  �  �  a    �  �  X  5  :  ?  A  <  6  +         �  �  �  ]    �  e  �  q  �  j  |  w  q  �  �  x  e  R  A  /    �  �  �  \     g  �  �      �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  v  h  Y  y  d  M  -  
  �  �  �  h  9  	  �  �  }  I    �  d  �  �  �  �  �    w  p  i  b  [  T  M  E  <  4  ,    �  �  y  I  %  &  (  +  -  ,  ,  ,  #    �  �  �  �  �    `  1    �  �  �  z  m  ^  L  7         �  �  �  z  G    �    I  �  �  �  �  �  {  e  K  B  @  &  
  �  �  �  z  M    �  y  �  �  �  �  �  �  z  d  M  5  !    �  �  �  �  �  �  d  �  �  l  S  5    �  �  �  z  ;  �  �  U    �  �  r    �  �  f  �  �  �  �  �  _  8    �  �  �  x  Z  `  y  �  �  o  H    �  �  &  4  7  3  )    �  �  �  m  5  �  �    _  �  �  w  �  �  �  �  �  �  q  ^  K  8  &      �    5  =  >  ?  ?  ,  {  �  '  \  z  g  0  �  D  �    P  �  z  �  �    	�  J  �  �  �  �  �  �  �  y  ^  ?    �  �  �  \    �  Q  �  �  �  �  �  �  s  Y  >  #    �  �  �  t  J    �  �  �  �  p    �  �  �  o  >    �  �  J    �  ~  8  �  �  c  *  
  