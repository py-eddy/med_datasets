CDF       
      obs    8   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type                     	amplitude                   comment       �Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM���   
add_offset               min       ?`bM���   max       ?���n��      �  �   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       N	L�   max       P�N�      �  �   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       �C�   max       >n�      �  l   effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>�z�G�   max       @E��
=q     �   L   effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ?���
=p    max       @vl�\)     �  )   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @$         max       @M�           p  1�   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @�e        max       @���          �  2<   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       ��h   max       >�o      �  3   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A���   max       B.t      �  3�   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A��0   max       B-�      �  4�   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       >���   max       C��      �  5�   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       >���   max       C�      �  6�   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max          �      �  7|   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min             max          ;      �  8\   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min             max          3      �  9<   
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       N	L�   max       PH�x      �  :   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6��C-   
add_offset               min       ?���D��   max       ?�s�PH      �  :�   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       �C�   max       >n�      �  ;�   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>��
=q   max       @E��
=q     �  <�   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ?���
=p    max       @vh�����     �  E|   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @         max       @M�           p  N<   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @�e        max       @�*           �  N�   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         C�   max         C�      �  O�   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6��C-   
add_offset               min       ?���Z��   max       ?�o���     �  Pl               _   _      	         +   �                  
      >   "                  6               w   '                  	      	   �   �      B         &               �         N�O#�$N�TN���P���O�-gN���N=�N�!rOF�IO�4P�N�O�#N�IN?�WNYU�N��OWG�P+U[PO�O���N	L�PRKO �DO��N�O��O1��O�(�P0Q#Nh/P�[�O�NN�$qO;�N*�N�]�N�~�Na,�O�%N���PX��POR�Px�O�YN`.O�z�NϲOl�OrH�N7�O���N��GNv��N-��C���9X���
�ě�;D��;D��;ě�;�`B;�`B<t�<t�<T��<�o<�o<�o<�1<���<�/<�/<�/<�/<�`B<�`B<�`B<�h<�h<�=+=C�=\)=\)=�P=49X=H�9=T��=T��=T��=T��=]/=]/=aG�=e`B=u=u=y�#=y�#=}�=}�=}�=}�=�C�=�O�=��
=��>�>n���������������������feggkt���������tsigfcgit{������������tgc������������������������/32@EFLOB5����������($#�����)67BFBB6)srt{�����tssssssssss�������������������������������������� ,<HUny�{ronaH<8-% )BNg���������g)meejmpxz�������{zmmm������������������������������������������������������������,'%(,-/6:<>A<96/,,,,����������������������������������������2+1AO[t����������hB2������
!
������� "/;?@;/,"          ����
)-+06?LJ5��%)*25565.)rt���������������xtr�������  ���������RRWmz����������zmXTR������!$����&"%)5N[fidd`[KB5331&����)?FE50/1/)��7?BN[]][NB7777777777zv{��������������z"/;HTempodH;"u{��������������{uuYUVaagmz�������zmcaY����������������������������������#(*'$#Y[bhtw���trh\[YYYYYY��������������������
#$-/233/)#����������������������������#*.,&�����*,/6<HLW]acja_UH<3/*����)59:2)!��������������I<800/0<ITQIIIIIIIII������������������������������������������������  ����������(8BO[ftrhf[JB6)�����������������������������
	�����"#/<<>><</+#""""""##"###
		
#####>BGHSUVUUH>>>>>>>>>>ĚĦĳĺĸĳĦĤĚĐĚĚĚĚĚĚĚĚĚĚ�zÇÓÖàìéÓ�z�o�n�e�d�a�_�a�a�n�t�z�5�<�B�L�N�V�R�K�F�B�5�)�!�'�)�/�0�4�4�5�����������������������������������������<�UŇŭžŹŭŔ�n�U�0�
�������������#�<��������� �������ּȼüƼͼּ�ÓàæààÖÔÓÉÇ�z�z�r�w�w�zÇÏÓÓ�������ûʻû����������������������������������������������������������������������(�5�?�>�>�5�(�������ݿٿݿ������������������������ùõù�������)�B�[�tđėēĆ�t�U�D�)�����
����)���(�5�=�A�N�Q�W�N�A�5�(��������ùȹϹܹ��ֹܹϹù¹��ùùùùùùùþʾ׾����׾ʾ��žʾʾʾʾʾʾʾʾʾʾs�������x�s�f�Z�b�f�o�s�s�s�s�s�s�s�s��"�/�;�H�T�V�T�H�F�;�/�%�"������������������������������������������������!�:�F�R�F�-���ֺ������������ɺֺ���f������ž¾���������f�[�R�W�E�@�=�S�f�Z�]�]�f�n�r�n�a�M�4�������A�M�S�Z�a�b�i�e�a�T�N�S�T�[�a�a�a�a�a�a�a�a�a�a��������;�P�k�i�T�H�/�"���������������������������������������������������z���������¿ÿƿ̿п˿Ŀ������������������������'�(�3�@�@�@�3�'����������āĚĦ������������ĳĦĚĉĀ�|�q�m�t�wā���������������������������������z�x��������� �%�%�#������ѿſ��������Ŀݿ�ƧƳ�������� ������ƧƎƁ�h�O�9�<�Z�uƧ���ĽŽǽƽĽ����������������������������������������������������s�Z�I�6�1�8�g���#�0�<�D�P�P�O�H�<�,�#��
���������
��#�������!�-�-�2�-�!��������������ŇŔŠŸŹ����������ŹŭŠŖŔŋŋŇņŇ�Z�f�j�h�f�d�Z�U�U�W�Z�Z�Z�Z�Z�Z�Z�Z�Z�Z�(�5�@�A�N�O�N�M�A�5�(� ����#�(�(�(�(�����ʾ׾��������׾ʾǾ��������������4�:�@�J�E�@�6�4�2�+�'�!�'�(�4�4�4�4�4�4ÇÓàãìíñóìâàÓÍÇÁÀÃÂÇÇ�`�m�y���z�y�m�h�`�T�G�=�G�K�T�[�`�`�`�`���@�Y�f�j�d�S�4�'����û��������ܻ��E\EiEuE�E�E�E�E�E�E�E�E�EuEiE_EZEPEEEVE\��(�4�A�S�S�M�A�4�+�(�����������¿��������������²¢£¬²¿�B�N�[�d�g�l�s�t�~�t�g�[�N�H�B�B�?�=�B�B�ֺкֺ����������ֺֺֺֺֺֺֺֺֽl�y�������������������y�l�`�T�R�S�d�`�l�������������������������}�v�����������������������������������������z�{�������������������������������������~�{�w�x�~��ù����������ýùìììóùùùùùùùùD�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D~DyD|D�D�ǈǔǡǬǭǡǡǔǈ�{�z�v�{ǆǈǈǈǈǈǈ�0�*�#� ����� �#�0�3�<�F�<�0�0�0�0�0ŭŹ��������ŹŭŬšŭŭŭŭŭŭŭŭŭŭ X @ n > R 5 j M Y g , _ H t O W i P ; Q H ] X ( 2 v 1 M I m v : # t 7 u 2 h B E j O 2 *   H 9 2 8 1 X P  E N G  A  �  p  �  �      -    �  �  O  =  c  E  b  �  �    �  �  G  �  ^    y  T  �  �  |  |  B  �  �  �  �  �  �  �  N  �  �  �  �  �  I  p    �  �    Y  E  �  j  ;��h;ě���C�;��
=Ƨ�=Ƨ�<T��<u<�j<u=T��>bM�<�h<�j<��
<�j<�=�P>�=�E�=u=o=m�h='�=8Q�=o=���=#�
=]/=�o=�P>�w=��=}�=�o=aG�=m�h=ix�=}�=���=�%>P�`>W
==��w>%=���=��w=���=��=� �=ě�=��>�o=�>�u>��B�1B
�B
�SB�~B.B�B1�B��B"�B�.BBʧA�j�B '�B!�7B��B;fB��B(�B��B#�GA���B0�B^Bg�B#�_B 2HB.tB�|Bj
B~B��A��/B)�{A�߷B�BO�BqB�B!�B��B*Bn�B'Bf�BB&i�B,^bB��B9�BS7B6�BїBF�B;�BU�B��B
 |B�B��B�3B�B�jB��B#<@BG9B��B�AA�r�B @wB!wB��BH�BB<�B��B$C?A��0B�SBjnBA B#G;B ��B-�B��B@�B��B?�A��B)��A�Y�B�QBA|B/�B�=B"?BZ�BHwB�1B9�B��B#B&E�B,@�BVbBHHBE�B7%B�?B>�B?�B=8A��A�*.A���A��A�+!AJ2A��@��@�A��rA��XA���A��>���ASACA��4A��@PTAEJ�A8A~A�3�A�5eA��At�?��A�F�Ar�7A�xB�A%IA���A�,i@d��A�!�A?I[A�w>AR �@�'�Aʓ�Ai�@���C��A6��A�AXA�h�@G��AJWA�>�A��+@��A��VC���B4�A��MA�/�A��4Aȇ?A��A�~A��tA�A� @�,�@�2�A��hA�w]AٔcA���>���AR�AC�KA��JA��@J�	AEC�A7��A��0A��}A���As�?��fA�s`AtʘA��B�CA%�A�n�A��@d�A�A>��A���ARM@͍�A�aaAkdr@�_[C�A6��A���A�x�@D�A�A�6}A�k�@��A͔C���BA�A�;A��               _   _      	         ,   �                        ?   "                  6               x   '                  	      	   �   �      B         '               �                        ;                  !   ;                     +   1   #      3            #         1      9                              3   '      )                                                 )                                             #         3                     1      +                              )         %                                 N�O;�N�c�N5�GP��OdI�N���N=�N�OF�IN�!�O�&3N���N�IN?�WNYU�N��OWG�O��O��yO�ҞN	L�PH�xO �DO^�N�O�#O1��O�(�P0Q#Nh/P&��O�	�N(�DO;�N*�N�]�N�~�Na,�O�%N���P�#Ow�qO?�$Ps�O�YN`.OF� N��Ol�OrH�N7�O(|N��GNv��N-�  x    �  �  -  �  5  �  1  -    `  3  �  =  �  2    �  8  V  �  a    d  e    d  I  �  �     �  (  �      �  B  {    �  *  �  L  7        �  �    /  �  �  ȽC���1���㻃o=t�=t�;ě�;�`B<o<t�=+>o<���<�o<�o<�1<���<�/=�{=@�<�<�`B<�h<�`B<��<�h=t�=+=C�=\)=\)=�+=8Q�=T��=T��=T��=T��=T��=]/=]/=aG�=�j=��#=y�#=�C�=y�#=}�=�C�=�%=}�=�C�=�O�>hs=��>�>n���������������������fegglt���������ttjgfot���������������zto�����������������������'1;;@?5)����������������)67BFBB6)srt{�����tssssssssss��������������������������������������//08<HKU^ZUMH<8/////@758>BN[gty���tg[NB@hglmtz������zmhhhhhh������������������������������������������������������������,'%(,-/6:<>A<96/,,,,����������������������������������������;;BO[ht��������thTG;������
 
����� "/;?@;/,"          ����),*06>KH5��%)*25565.){vy����������������{�������  ���������[XW\mt����������zma[������!$����&"%)5N[fidd`[KB5331&����)?FE50/1/)��7?BN[]][NB7777777777��������������������"/;HNTcjpnbH;"��������������������YUVaagmz�������zmcaY����������������������������������#(*'$#Y[bhtw���trh\[YYYYYY��������������������
#$-/233/)#��������������������������
!"
�����+,/7<HKU\bca]UOH<6/+�����)2787.)�������������I<800/0<ITQIIIIIIIII������������������������������������������������  ����������(8BO[ftrhf[JB6)���������������������������

��������"#/<<>><</+#""""""##"###
		
#####>BGHSUVUUH>>>>>>>>>>ĚĦĳĺĸĳĦĤĚĐĚĚĚĚĚĚĚĚĚĚ�zÇÓÔàëèàÓÇ�z�n�e�a�`�a�a�n�v�z�B�G�K�N�P�H�D�B�:�5�.�)�'�)�+�1�3�5�;�B�����������������������������������������I�U�nŇŋŁ�n�U�0�#�
������������#�<�I���������������ּԼϼϼҼּݼ�ÓàæààÖÔÓÉÇ�z�z�r�w�w�zÇÏÓÓ�������ûʻû����������������������������������������������������������������������(�5�?�>�>�5�(�������ݿٿݿ�������������� �������������������������6�B�O�[�h�l�v�y�v�m�h�[�O�B�6�2�1�1�4�6��(�5�8�A�J�A�A�5�(�����������ùȹϹܹ��ֹܹϹù¹��ùùùùùùùþʾ׾����׾ʾ��žʾʾʾʾʾʾʾʾʾʾs�������x�s�f�Z�b�f�o�s�s�s�s�s�s�s�s��"�/�;�H�T�V�T�H�F�;�/�%�"������������������������������������������������������!�'�&������ֺɺǺĺʺֺ�����������������������s�f�Q�M�O�Z�f��4�A�M�Z�i�n�k�]�M�A�4�(���������(�4�a�b�i�e�a�T�N�S�T�[�a�a�a�a�a�a�a�a�a�a��������;�N�e�_�T�H�/�"���������������������������������������������������z���������������Ŀɿ˿Ŀ������������������������'�(�3�@�@�@�3�'����������āčĚĦ��������������ĳĦĚčă�v�u�{ā���������������������������������z�x��������� �%�%�#������ѿſ��������Ŀݿ�ƧƳ�������� ������ƧƎƁ�h�O�9�<�Z�uƧ���ĽŽǽƽĽ����������������������������������������������������s�g�V�I�A�?�L����#�0�B�N�O�M�I�G�<�0�#��
���������
�����!�)�,�!����� ��������ŇŔŠŸŹ����������ŹŭŠŖŔŋŋŇņŇ�Z�f�j�h�f�d�Z�U�U�W�Z�Z�Z�Z�Z�Z�Z�Z�Z�Z�(�5�@�A�N�O�N�M�A�5�(� ����#�(�(�(�(�����ʾ׾��������׾ʾǾ��������������4�:�@�J�E�@�6�4�2�+�'�!�'�(�4�4�4�4�4�4ÇÓàãìíñóìâàÓÍÇÁÀÃÂÇÇ�`�m�y���z�y�m�h�`�T�G�=�G�K�T�[�`�`�`�`���@�V�V�P�4�"�����ܻлĻ»Ȼ����E�E�E�E�E�E�E�E�E�E�E�E�EuEnEjElEpEuE}E���(�1�A�Q�Q�M�A�4�(������ �����¿�������	������������²ª¦°¿�B�N�[�d�g�l�s�t�~�t�g�[�N�H�B�B�?�=�B�B�ֺкֺ����������ֺֺֺֺֺֺֺֺֽl�y�������������������y�l�c�[�V�X�`�i�l�������������������������x�������������������������������������������z�{�������������������������������������~�{�w�x�~��ù����������ýùìììóùùùùùùùùD�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�ǈǔǡǬǭǡǡǔǈ�{�z�v�{ǆǈǈǈǈǈǈ�0�*�#� ����� �#�0�3�<�F�<�0�0�0�0�0ŭŹ��������ŹŭŬšŭŭŭŭŭŭŭŭŭŭ X ? i G U ' j M X g , ' : t O W i P 5 1 H ] U ( 1 v , M I m v - # d 7 u 2 h B E j @  '  H 9 * 0 1 X P  E N G  A  W  �  j  �  �    -  �  �  �    �  c  E  b  �  �  
  �  \  G  �  ^  �  y  �  �  �  |  |  �  �  d  �  �  �  �  �  N  �  �  �  �  G  I  p  �  �  �    Y  a  �  j  ;  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  x  g  U  D  2       �  �  �  �  �  �  �  }  f  B     �   �      �  �  �  �  p  @  
  �  �  a  (  �  �  y  B    �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �    ~  |  {  W  ]  g  }  �  �  �  �  q  ^  F  #  �  �  �  r  A    �  �    c  �  �       (  ,  #    $  %  �  �  �  Y  �  �  �  9  
  
�    ]  �  �  �  �  �  �  o    
�  
  	^  �  �  �  �  `  5  *       �  �  �  �  �  o  >    �  �  �  �  �  �  �  �  �  �  �  �  �      +  3  8  >  E  K  P  U  Z  _  d  i  o  .  0  /  -  (             �  �  �  �  i  L  /    �  �  -        �  �  �  �  �  �  �  �  j  O  9  "  	   �   �   �  �  �  �  �  &  k  �  �  �        �  �  N  �  �  +  �  5  �    �  -  }  �    �  �  ,  [  U    �  �  �  �  �  	'  �  ,  ,  )  &  0  ,      �  �  �  �  h  <     �  d    �  2  �  �  �  �  �  �  �  q  ^  L  :  )      �  �  �  �  �  �  =  :  7  4  1  .  *  &  #              �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  y  k  ]  O  A  2  $  2  +  #      %  2  ?  B  5  )      �  �  �  �  �  �  �    �  �  �  �  �  �  �  �  �  ~  d  J  /    �  �  �  o  6  
L  F  �  �  �  8  a  o  }  �  m  .  �  B  �  
�  	�  M  �  [  �  &  �  �    +  8  2       �  �  }  D  �  �    �  d  U  �  B  U  N  C  5  '      �  �  �  �  <  �  �    �  '  �  �  �  w  m  c  Z  S  M  G  @  4  "    �  �  �  �  �  }  `  `  \  J  +    �  �  �  ^  6    �  �  �  �  m    �  v  �    �  �  �  �  �  �  �  ~  l  \  K  :  +         �  �  +  `  c  d  a  \  U  K  =  ,    �  �  �  �  b  0  �  �  Z    e  ]  U  M  E  <  4  0  -  +  (  &  #    �  �  �  �  �  v  �        �  �  �  �  �  R  
  �  e  
  �  A  �    B  �  d  V  I  ;  +      �  �  �  �  �  �  �    l  X  B  ,    I  ?  7  6  0  #      �  �  �  �  T  #  �  �  ~  :  �  {  �  �  �  �  y  Z  =    �  �  �  H    �  �  �  �  t    �  �  �  �  �  x  o  g  _  V  N  >  &     �   �   �   �   �   ~   f  	�  
g  
�  
�  
�  
�  
�  
�  
_  
  	�  	(  �  C  �  '  n  k  %  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  4  �  ~    �    �  �  �  	      �  �  �  l  e  V  "  �  �  r  5  �  �  �  v  _  G  -    �  �  �  �  �  �  s  ]  D  $  �  �  �  �      �  �  �  �  �  �  �  d  C  #     �   �   �   �   �   �   �    �  �  �  �  �  �  �  �  �  �  �  �  �  L    �  �  �  h  �  |  o  b  T  C  3  #    �  �  �  �  �  g  B     �   �   �  B  2  #    �  �  �  �  �  t  i  g  e  e  d  c  a  ^  X  S  {  d  H  (     �  �  D  �  �  D  �  �  Q    �  }  $  �  |    �  �  �  �  �  �  �  �  i  O  5      �  �  �  �  %  �  m    �  �  �  �  �  |  $  �  E  �  >  y  �  
�  	9  �  �  �  �  e  �  x    �  �  !  )    �  i  �  )  9     F  
?  �  f  �  �  �  �  �  �  �  �  �    j  S  6    �  �  �  �  �  �  +  A  L  F  ;  +    �  �  �  J  
  �  �  2  �  3  S  T  �  7  )      �  �  �  �  u  T  0  	  �  �  n  /  �  �  n      �  �  �  �  t  Q  %  �  �  �  m    �  �  1  �  ~      �  �    
      
        �  �  �  �  �  o  /  �  �  �  �       
       �  �  �  �  �  �  �    c  C    �  �  s  $  �  �  �  y  f  O  5    �  �  �  �  k  =    �  �  
  t  �  �  �  �  �    R  "  �  �  m    �  }  1  �  �  %  �  �   �      �  �  �  �  �  �  �  �  �  �  s  d  U  M  I  D  ?  ;  �  �  S  �  �    '  +    �  '  y  �  �  v  �  Q  �  
�  �  �  �  �  �  y  V  .    �  �  }  O  !  �  �  ~  �  �  F  �  �  W    �  �  H    �  [    �  �  N    �  j    �  r    �  �  �  �  |  h  U  B  /      �  �  �  �  �  �  k  S  <