CDF       
      obs    5   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type                     	amplitude                   comment       �Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM���   
add_offset               min       ?h�t�j~�   max       ?�x���F      �  �   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       NR	   max       P��>      �  �   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       �ě�   max       =��      �  T   effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>޸Q�   max       @E�(�\     H   (   effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ?�Q���   max       @vt�\)     H  (p   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @)         max       @L�           l  0�   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @�        max       @�Z�          �  1$   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       ;o   max       >`A�      �  1�   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A��&   max       B,��      �  2�   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A�x   max       B,�1      �  3�   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       ?���   max       C�m'      �  4t   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       ?�N�   max       C�[      �  5H   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max          �      �  6   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min             max          I      �  6�   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min             max          G      �  7�   
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       NR	   max       P�9�      �  8�   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6��C-   
add_offset               min       ?�`�d��8   max       ?�t�j~�      �  9l   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       ���
   max       >1'      �  :@   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>�ffffg   max       @E�\(�     H  ;   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ?��\)    max       @vt�\)     H  C\   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @         max       @L�           l  K�   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @�        max       @���          �  L   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         F�   max         F�      �  L�   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6��C-   
add_offset               min       ?��s�g�   max       ?⒣S&     �  M�                  4   2            w         B   B   �      :      W   
   .      B         	   '   c   	      :                  /   %         h            
   /            �      kN�	~NV�N��=OEhnOv�OԨ�O�X�O��N�@tO�f}P+�O,߉Ov$O��HP'�}PP!�N?i�P��N�8�P��>Nu(&P!QIN]��P0�N���N_�N�]O�:�P�}�N %�O�-O��N9$CN�>�Oc�N�_�N�LO���O�~ZOQ-�N:��O���NU�=O��N�fN���O��eNR	OzMNcìO�7�NA�O��B�ě����
���
:�o:�o;D��;��
<#�
<T��<e`B<u<�o<�o<�o<�C�<�t�<���<��
<��
<�j<�j<ě�<ě�<ě�<�/<�`B<�<��=o=o=C�=C�=C�=��=�w=,1=,1=49X=8Q�=8Q�=<j=D��=H�9=L��=P�`=Y�=]/=ix�=m�h=m�h=}�=�\)=������������������������QMNQUbnvzzunlcb`XUQQ��������������������������������������������6<@DB=6)��`XUV]dnz���������zn`���������
""##
���"/6;?@;;/"��������������������`djz�������������tl`��
#/773# 

��X[bhmt~���������th[X�����������������������)BHKIG@5)����>?J[g����������g[ND>MEENT[^a][VNMMMMMMMMhhppt�������������th������������������������5PZ_qf[NB-����v������������������
"&'�����5366BHNOSOEB=6555555::GP[otz������thOB:��������	�������DOW[hihh[ODDDDDDDDDD��������������������gbbdhjoz~��������zpg��������&4=8)������������������������7;=9;HTamz����zdXTH7 ��
#/?DHHLURH8#
 "#+/0<<=?@=<10.#""""�����
��������������wtnjpt��������bdcfmnz�������znbbbb��������������������������
/<BHONH/#
�����������������������KFEN[gt�����tkg][NK����������������������������������������BBO[dhhmjh[ODBBBBBBB��������������������436BOOOIDB?644444444 #*/<=@<3/#��������������������;:<BHKTNH<;;;;;;;;;;�����)--)#�����������
����)./,)�������
"%%
���⺗��������������������������������������ìù������������ûùìëìììììììì�x�������������������x�p�l�_�^�_�l�p�x�x�����ʼ�������ּʼ������������������ûǻлۻջлǻû����������������������þ��������ɾ�������s�Z�M�A�6�4�>�M�Z���E�E�E�E�E�E�FF	FFE�E�E�E�E�E�E�E�E�E���"�/�;�H�T�a�m�s�{�m�a�T�;�"������T�a�m�o�z�����z�u�m�a�U�T�O�K�P�T�T�T�T��4�M�i�q�r�h�Z�P�M�A�4�(����������л������������ܻ��������������������������������������������z�s�g�b�s�z���@�E�Y�_�e�m�r�t�r�g�e�Y�L�H�@�9�2�/�3�@ù����������������ìßÓÇÄ�~�{�ÌÓù�b�h�W�Z�vÁÈ�z�n�U�������� ��6�:�H�b��6�O�d�l�s�n�f�[�6���������������������������������������������������������������!�1�:�>�8�0�"�������������������"�.�4�0�.�%�"��	��	�
������������$�9�=�����Ƴ�h�6�&�� �O�g�~ƋƳ���/�<�E�H�K�H�A�<�/�&�$�.�/�/�/�/�/�/�/�/������B�N�g�{�o�a�U�B�5�������������Ѽ��������Ƽ��������������������������������'�M�r���������r�Y�@�4���ܻ˻ɻлٻ����������ʾ̾ʾž������������������������H�N�U�Y�W�U�H�F�?�<�H�H�H�H�H�H�H�H�H�H�����������������z�y�t�v�z��������������������!�-�D�F�R�-�!����ۺ̺ͺֺ���0�bŇœŜŝŘŎ�n�U�0���������������²¿����¿²¦¡¦­²²²²²²²²²²ĳľ���������	�������������ĿĳĬĞĳ�����ʾ׾��ܾʾ������s�f�Z�Q�f�o�{�����Y�\�f�r�s����r�f�Y�X�M�H�M�W�Y�Y�Y�Y�;�G�T�W�`�m�`�T�G�;�.�,�.�1�;�;�;�;�;�;�����������������������������������������@�L�Y�e�r�s�x�r�j�e�Y�L�F�@�:�8�@�@�@�@�Y�]�b�]�Y�Q�M�@�@�@�M�P�Y�Y�Y�Y�Y�Y�Y�Y�G�T�`�m�y��������y�m�m�`�Z�O�F�<�5�>�G����������������ùý�����������m�y���������������y�m�`�T�N�T�V�`�b�i�m�(�4�7�A�D�A�A�4�+�(��"�(�(�(�(�(�(�(�(�ʼ�����	�������ּʼ������������ʻ:�:�>�E�F�G�F�:�-�,�+�'�-�6�:�:�:�:�:�:�y�������������������y�s�l�f�c�c�l�u�y�y���������������������������������
���������������������������/�;�T�m�w�����������z�a�H�;�/�	������/���������������������������������������ҿ��������ÿпѿٿѿɿ������������~������������������������������D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�DxDD�D�����������������������������������������EiEuE�E�E�E�E�E�E�E�E�E�EuEmEiEgEgEfEeEi = Y * D 6 Z 0 9 & 7  m B ; P  p Q L U A a 5 _ H s O Q ? c Z Z ^ b 6 + : H G 3 8 2 f 1 @ K e 0  I 2 >      �  R    �  A  P  ^  �    �  �  �  K  �  :  _  G    �  o  s  m  a  �  �  s  �  |  �  0  �  _  �  �  E  
  @  �  b  �  Y  �  �  7  =  �  a    �  |  ~  F  0;�`B;ě�;o<��
<�o=]/=]/<�/=+=#�
>1'<�/=@�=��T=��>]/<ě�=��w<ě�=�;d=o=�O�<�=�E�=\)=o=��=�\)>$�='�=q��=�^5=��=0 �=P�`=y�#=D��=�Q�=��T=��=H�9>��=e`B=�C�=aG�=�%=��=}�=� �=}�>`A�=���>@�B"/�B�B'ϼB!��B"��B�[B^BB��A��&B �B�Bk2B` B!��B�UB	ǂB�+B
�yB�sB�TBO�B��B@'B��B#�GB�B�B��B�}B�A��B��B%��B�KBvBoB6BP3B��B	d�B9OBx\BJB,��B1�BԙB3�B!�B��B�/B`B��BoGB">�B< B'��B!��B"��B�,B2|B��A�xBſB:|B��B��B"<HB�B	�\B��B
��BǼB;nB|?BY�B?BeB#�\BǄB�B��B��BĉA��tB��B%��B��B�B>3B?�B' B��B	j�B��BEBDZB,�1B5lB?�B��B?�BhbB�EB;mB B�[@t�A��U@�L"@�N@�SXAA�C�m'A���A��1A9�s@�UIA�W�?���A�f�A�u�A�p3A�NaA��PA]�B�A��A��}@��@@�'�AK�AĶNA���@^�A�^A��LA���ALU<@��_Aej�As�l?ճp@���Ai|A�kAm�A8��A��@{b�A�@W�A���A��6A�WAsԦA���C���@AC�	�@�A΁'@�!�@��@��AA)C�[A�n0A���A:�@� �A���?��$A�}�A�y�A�zyA��[A��8A]�nBmA��A�s�@��T@���AK�AĂ�A��|@]>�A�|�A�}�A�u&AL��@�0�Ae8At�?�N�@�.�Aj�aAс�Am�A9�A�@|eA�@Zo�AҀA�z�A��As <A��\C���@�"C�W                  4   2            w          B   B   �      ;      W   
   .      B   	      	   (   d   
      :                  /   %         i               0            �      k                  '      !         +         !   1   -      +      I      1      1               7      '   '                              !               )                                                                              G      '      +               1                                                                        N���NV�N��=N�%YOv�O1B�O�X�O�LN�^N�K�Oq�!O,߉N;�O6�	O�]�O���N?i�O�)�N�8�P�9�N,�O�qN]��P�yN���N_�N�]O�:�Pd�sN %�O*�Og��N9$CN�>�Oc�N�k�N�LO���OeE�NM�qN:��O\¡NU�=N�!�N�fN���O���NR	O.I�NcìOpNA�Oj��  ?    �  U  _  �  �  �  �  �    '  �  �  �  �    M  �  �  B    x  ^  �  ^  #  �  	�  �  �  9  /    �  9  �  X  h  �  �  T  �  �  �  O  �  �  �  C  �  3  a���
���
���
;ě�:�o<�j;��
<49X<�C�<�/=�C�<�o<�9X=��=t�=���<���=8Q�<��
<ě�<���<��<ě�=\)<�/<�`B<�<��=@�=o=0 �=Y�=C�=��=�w=<j=,1=8Q�=L��=e`B=<j=�-=H�9=aG�=P�`=Y�=��=ix�=�o=m�h>1'=�\)=Ƨ�����������������������QMNQUbnvzzunlcb`XUQQ����������������������������������������
	)0669864)`XUV]dnz���������zn`�������
 !  
����"(/0;<=;//"������������������������������������������
#/773# 

��g`cghqt��������tihgg��������������������)5?BBBB>95) SQRW_gt���������tg[SMEENT[^a][VNMMMMMMMMssy���������������ws�������������������������5OZgqeNB-�����������������������
"#
 ����5366BHNOSOEB=6555555=?HPU[ht~}����zthOB=��������	�������DOW[hihh[ODDDDDDDDDD��������������������gbbdhjoz~��������zpg�������$.35)������������������������GEBBEHRTamuytmha[THG
#/4<?@>;/#
"#+/0<<=?@=<10.#""""�����
��������������wtnjpt��������kgfintz}�����}znkkkk��������������������������
/<BHMH/#
�����������������������Y[cgt{~vtgc[YYYYYYYY����������������������������������������BBO[dhhmjh[ODBBBBBBB��������������������436BOOOIDB?644444444 #*/<=@<3/#��������������������;:<BHKTNH<;;;;;;;;;;�����!&'#���������

������)./,)������
""
����㺗��������������������������������������ìù������������ûùìëìììììììì�x�������������������x�p�l�_�^�_�l�p�x�x�������ʼּ޼ּռ̼ʼ��������������������ûǻлۻջлǻû����������������������þM�Z�f�s�����������s�s�f�Z�R�M�C�D�M�ME�E�E�E�E�E�FF	FFE�E�E�E�E�E�E�E�E�E��/�;�H�T�a�o�v�y�p�a�T�;�"���
����/�T�a�i�m�y�z�|�z�o�m�a�[�T�T�O�S�T�T�T�T�(�4�A�D�M�Z�Y�M�E�A�4�(�%�����&�(�(���ûлܻ���������ܻл»����������������������������������������z�s�g�b�s�z���3�@�L�Y�Z�e�i�o�e�`�Y�N�L�@�=�7�3�3�3�3ìù������������ùìàÓÌÇÄÇÊÓàì�/�<�H�U�a�i�t�w�n�a�U�H�<��
���
��/���6�B�O�V�Z�W�P�B�6�)����������������������������������������������������������������	�������������������˿�"�.�4�0�.�%�"��	��	�
��������������$�0�;�����Ǝ�h�6� �"�O�hƁƍƳ���<�=�H�I�H�?�<�/�(�&�/�:�<�<�<�<�<�<�<�<������)�B�[�c�f�Z�M�B�5�������������Ѽ��������Ƽ��������������������������������'�@�M�f�v�����r�Y�@�4����ܻӻϻֻ�����������ʾ̾ʾž������������������������H�N�U�Y�W�U�H�F�?�<�H�H�H�H�H�H�H�H�H�H�����������������z�y�t�v�z��������������������!�-�D�F�R�-�!����ۺ̺ͺֺ���0�IŇŒŕŐń�n�I�0��������������
�²¿����¿²¦¡¦­²²²²²²²²²²Ŀ������������������������������ĿĺĽĿ�������ʾξ׾ؾоʾ���������������������Y�\�f�r�s����r�f�Y�X�M�H�M�W�Y�Y�Y�Y�;�G�T�W�`�m�`�T�G�;�.�,�.�1�;�;�;�;�;�;�����������������������������������������@�L�Y�e�m�r�t�r�e�c�Y�L�J�@�=�>�@�@�@�@�Y�]�b�]�Y�Q�M�@�@�@�M�P�Y�Y�Y�Y�Y�Y�Y�Y�G�T�`�m�y����������y�m�`�[�P�G�=�6�?�G���������������������þ����������y�|���������y�m�i�i�m�w�y�y�y�y�y�y�y�y�(�4�7�A�D�A�A�4�+�(��"�(�(�(�(�(�(�(�(�ʼּ�������������ּɼ������������ʻ:�:�>�E�F�G�F�:�-�,�+�'�-�6�:�:�:�:�:�:�����������������y�l�k�j�k�l�y������������������������������������������
���������������������������;�H�T�a�q����������z�m�a�B�;�/�*�#�)�;���������������������������������������ҿ����������ĿȿĿ�����������������������������������������������D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�����������������������������������������E�E�E�E�E�E�E�E�E�E�E�E�EuEqEjEjEkEuE|E� ? Y * 4 6 = 0 8 3   m E 2 <  p H L R : ` 5 ^ H s O Q @ c + , ^ b 6 ) : E 0 ; 8 ) f ; @ K G 0  I  >     |  R      A  w  ^  s  �  �  �  �  �  �  �  d  G  (  �  6  T  �  a  �  �  s  �  |  �  0  k  �  �  �  E  �  @  �  �  p  Y  �  �  �  =  �  0    n  |    F  �  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  4  <  ?  =  6  *    
  �  �  �  ~  V  /  
  �  �  �  	           �  �  �  �  �  t  a  W  0  �  �  �  I    �  �  D  �  �  �  �  �  �  }  w  r  e  X  J  :  )       �   �   �   �    1  =  R  U  U  R  L  D  6  #    �  �  �  �    Z  2    _  Y  R  J  @  5  &    �  �  �  �  x  O    �  �  R    �  �  .  R  s  �  �  �  �  �  �  e  ;  
  �  �  &  �  �  �  f  �  �  �  �  �  v  =  �  �  �  �  w  @  �  �  �  V  �  �  �  �  �  �  �  �  �  �  �  �  �  u  c  O  7      �  �  �  �  �  �  �  �  �  �  �  �  v  S  /  
  �  �  �  �  x    �  �  �    0  C  O  Z  h  y  �  �  �  �  g  G  #  �  �  �  0  �  �  �  	�  
g  
�  a  �  �  �  �  �  �  =  
�  
L  	�  �  U  �  M  '      �  �  �  �  �  �  �  �  �  �  �  �  �  o  R  U  d  �  �  �  �  �  �  �  �  �  �  �  �  n  1  �  �  g     �  �  u  �  -  {  �  �  �  �  �  �  z  D  �  �  :  �  7  �  �  F  }  �  �  }  �  �  �  �  �  �  �  ^  /  �  �    ,    �   �  D  �  �  �  N  �  ]  �  �  �  y  #  �  �  �  �  f    �  �      �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  _  -  >  n  �  �  �    3  F  L  F  '  �  �  R  �  �        /  �  �  �  �  �  �  �  �  �  �  �  y  o  e  \  S  K  C  ;  3  �  �  �  �  {  N    �  �  �  �  �  �  ]  3  �  �    S  U  #  .  :  >  A  8  *      �  �  �  �  �  u  V  5    �  �  �  �         �  �  �  �  �  �  }  @  �  �  i  �  o  �    x  u  r  o  j  d  ^  [  Y  X  W  Y  Z  ^  e  m  u  ~  �  �  �    J  ]  P  4    �  �  0  >  0  �  �  =  �  -  {     �  �  �  �  �  �  �  �  �    n  ]  J  7  %      �  �  �    ^  S  H  =  2  '      	  �  �  �  �  �  �  �  �  �  �  �  #            �  �  �  �  �  �  �  }  h  Q  9    �  �  �  ~  ]  @  %  	  �  �  �  t  A    �  [  �  2  a  �  �   �  	p  	�  	�  	�  	�  	�  	�  	c  	:  	  �  �    �  '  �  �  �  �  �  �  �  �  �  �  
            	     �  �  �  �  �  �  �  I  D  S  �  �  �  �  �  �  �  �  �  �  [    �  h  �  A   �  �  i  �  �    )  7  3    �  �  �  -  �  [  �  M  �    �  /  '           �  �  �  �  �  �  �  m  N  ;  .                 �  �  �  �  �  �  �    f  N  6    	   �   �   �  �  �  �  �  �  �  �  q  _  H  ,    �  �  �  g  9    �  �  &  .  4  7  9  8  2  #    �  �  �  c    �  V  �  q  �  Z  �  �  �  �  ~  q  c  R  ?  ,      �  �  �  �  �  �  i  R  L  K  1         �  �  �  s  5  �  �  (  �  a  �  )  �  �    I  f  e  T  ;    �  �  \    �  �  =  �  �  &  c  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  j  (  �  �  �  �  �  �  �  �  �  �  �    }  {  z  x  }  �  �  �  �  �  �  5  �  x  �    5  L  T  C    �  _  �  Q  
�  	�  �  e  �  #  �  �  �  �  �  �  �  �  �  }  t  j  T  >  $    �  �  �  �  �  �  �  �  �  �  �  �  �  �  �    [  4    �  �  r  �    �  �  �  �  �    �  �  �  �  �  �  �  �  �  �  �  �  �  �  O  >  ,       �  �  �    T  )  �  �  �  ^  $  �  �  ?   �  _  �  �  �  �  �  �  �  �  �  �  [    �  V  �  v  �  ]  l  �  �  �  �  �  �  �  �  _  6    �  �  �  ]  /   �   �   �   j  �  �  �  �  �  �  �  �  �  �  k  J    �  �  N  �  j  �  �  C  F  I  L  N  M  B  7  ,  !      �  �  �  �  �  �  �  �  �  �  @  �  �     X  �  �  �  �    i  z  b  >    l  l  
�  3  -  '           �  �  �  �  �  �  �  �  �  x  f  T  B  �    \  `  J  "  �  �  U  �  �  �  9  Q  6  �  
�  	\  �  f