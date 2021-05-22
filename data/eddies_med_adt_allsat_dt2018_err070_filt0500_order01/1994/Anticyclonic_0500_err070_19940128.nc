CDF       
      obs    5   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type                     	amplitude                   comment       �Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM���   
add_offset               min       ?h�t�j~�   max       ?���"��`      �  �   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       M��   max       P\��      �  �   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       ��o   max       =��      �  T   effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>��Q�   max       @F}p��
>     H   (   effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ����P    max       @vv�Q�     H  (p   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @,         max       @P�           l  0�   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @ʐ        max       @�          �  1$   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       ��`B   max       ><j      �  1�   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A�Z0   max       B2|�      �  2�   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A��   max       B22Z      �  3�   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       ?��   max       C�jj      �  4t   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       >�FT   max       C�h       �  5H   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max          z      �  6   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min             max          3      �  6�   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min             max          -      �  7�   
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       M��   max       P.��      �  8�   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6��C-   
add_offset               min       ?���D��   max       ?��/��w      �  9l   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       ��o   max       =�F      �  :@   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>�\(�   max       @F}p��
>     H  ;   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ?��Q�     max       @vp��
=p     H  C\   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @+         max       @P�           l  K�   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @ʐ        max       @�`          �  L   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         >�   max         >�      �  L�   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6��C-   
add_offset               min       ?��-V   max       ?���>BZ�     �  M�   	               (   5   #            d                                    )   
   !         "      .            	      
       *               	                  y      C   !NG5�O�r*N+�7NA�4NB-�P8 P�zO�*N��GN�e�N��O�ȀOIN�$N�BcN�;RN��Nվ�N��N�d�O��DN%�oO ~O��NꩱO�'O�,tO�AP\��NQ/P=>�P��N)�OӄOO��O�_7O��O��O�$	O?3�O3��O<%8N��	O�M��N9��O�AOU�N(��O��NYu�O�͓O��o�D���o:�o:�o:�o:�o;�o;ě�;ě�;�`B<o<o<t�<49X<49X<D��<D��<T��<e`B<�o<�o<�C�<���<��
<�1<�9X<�9X<�9X<�9X<�j<ě�<ě�<�/<�h<�h=+=C�=C�=t�=�P=��=��=49X=@�=@�=D��=aG�=ix�=�%=�+=�{=��jhrt�����tjjjjjjjjjjZY[ac`g���������tgbZ��������������lrt��������tllllllll��������������������������
/<JJ5.+#��326BO[ht������|hOBB3~������������������~������������������������������������������������������������{{~����������������{��������������������HHKT^aeea\TTMHHHHHHH��������������������
#/2<>>=<0/-#wz���������zwwwwwwww<64<>EHKUVWZafaUH<<<����������������������������������������-)+*/?BNT[`gjjg[NB5-xxz�������}zxxxxxxxx369BO[chhihe[QOOC;63"/<HUak^WU_^UH<5##.005<EHE<;0)$�������������������������������������������""������������������#*/0/,#��)5?PSN5���������75)�����������������������������5HHN[XE5)������������������������������������������/159BBN[\cb][TNDB=5/	
	"/7;<83/"			#)5BNNOIB5)	����
!#(/001/&#
��#).0<LWMIC<40#24127<HU_adgea^UHB<2��������������������MMO[\hu~�������uh\ZM��������������������788BOUOOHB7777777777���������������������������� �������A=@BO[c[WOCBAAAAAAAA����
����������dnp{��������{nddddddz|����������������{zyiffkms������������y�L�Y�e�j�m�e�Y�L�L�I�L�L�L�L�L�L�L�L�L�LÇàìù������������ùìÓÇ�z�i�a�e�nÇ����������ùõñùú�������������������ſy����������������y�s�m�y�y�y�y�y�y�y�y�G�T�V�Y�W�T�G�F�<�A�G�G�G�G�G�G�G�G�G�G�T�`�y�������������y�`�G�.�"�������.�T�����ͼڼܼ�߼ռм��������y�r�`�[�d�r���/�;�H�L�J�J�O�G�8�/�"�	�������������/����������������������������������������������	��	����������������������������#�/�<�H�O�I�H�>�<�1�/�$�#�������O�[�h�t�z��y�[�B�7�)�%������%�6�O��������+�-�)�'������ܻһܻ��������������ݿѿοѿܿݿ���������������������¿ĿѿӿҿѿϿĿ�����������������������������������s�q�f�e�f�l�s�������āăĆčĚĚĚčā�|�|�~āāāāāāāā�a�m�z���������������z�m�d�a�Y�U�Y�a�a�aE�E�E�E�E�E�FFFE�E�E�E�E�E�E�E�E�E�E���������� �������������������������������������$�;�.�$���
������������ƿ���A�N�W�R�N�A�5�(�(�(�5�9�A�A�A�A�A�A�A�A�׾پ�������ھ׾ʾľ������������ʾ־������������������ùìåêõ�����뼋��������������������r�f�f�`�f�r������Ϲ��'�1�8�3�'������ܹù����������tāčĠĦįĵĿ��ĳĦčā�t�h�O�V�^�n�t�u�v�uƁƃƁƄƆƁ�u�h�a�\�V�P�L�O�\�h�u������B�j�k�b�N�B�5�)����������������������������������������������������������<�\�{ŌŕŚŊ�b�<�0����	����������<�ݿ����ݿѿĿ����y�T�A�;�F�Q�m�����ѿݼ������ʼμʼż��������������������������f�s�u�t�x�x�u�Z�M�F�A�4����'�/�A�M�f�����ܹٹֹ۹޹��������������ݽ���������ݽĽ����������������Ľݿ�������)�*�(����������������������"�/�A�T�Y�[�L�;�/�"�	������������)�5�N�[�l�t�~�t�[�N�5�%�������(�5�6�>�H�N�P�N�I�A�5������	���ּ�����������������ּѼ˼ϼ��s�������������������������w�s�l�f�g�h�sùþ��������ýùìàÓÒÓÛàâì÷ùù����	�������	������������𾌾��������������������������������������ܻ����� �����ֻܻܻܻܻܻܻܻܻܻܻܽ����������y�s�g�p�y�����������������������������������������������������������������������Ⱥ���������������������������D�D�D�D�D�D�D�D�D�D}DuDuDzD�D�D�D�D�D�D�����������������������������������Ҽ��!�:�@�?�.���ּ˼������������zÇÓàùý����������ùìàÛÓÎÇ�{�z I \ | k U 9 < > @ T 6 6 , � K ? � D c > : c % F 2 A 7 ] : ? < r 5 V 6 e L S J ) b  E S 6 O [ / f I q n '    l  �  �  �  �  T  �  7  �  �      �  �  �  �  �    %  �  k  E  )  �      ]  |  �  W  ~  U  C  !  4  �  /    �  �  �  �  �  \    `  6  �  n  �  �  �  3��`B<49X<D��;��
;�o=#�
=Y�=��<#�
<49X<�j=�/<ě�<49X<�C�<�o<�o<�C�<�h<�`B=+<�9X<�h=q��<�=]/=8Q�<�/=aG�<�h=�C�=H�9<�/=D��=�P=Y�=,1=��=���=e`B=aG�=]/=]/=T��=Y�=]/=u=�t�=u><j=��P>��>1'Bd�B	�DB��B
!�B�B6�B�XB��BeB!�B� B��B!� A�bB �B�	B W�Bp�Bh�B�UBgB��B�yB��B%ӸB�BâBYB��B�@B��BM�B"�XBf�B!2�B!!�B�A�Z0B��BW�B%a�BT�B!�\B2|�B�BGB,I�B �B�B�B(�B��B _aBLYB	��B��B	�zB:�B1�B��B�B�DBA�B�TB�DB!�cA�ibB �B<?B ��B:�BAB­B6B��B 6B�B%�Bb�B��B;BA*BC�B�BZ�B"�BQ!B!/B!@,B�ZA��B�|BC[B%AoB��B!�yB22ZB>B;�B,K�B<�B]tB��B) �B@�B
&?�A�z�AΊ�AnyjAe�CAjNS@�X�A��A���A�/�Aµ�Aص@�ʮA}�oAxj�AE�A�1A��C�jjAШKBүA�T�AR�A�Q	@�"�?��A���Bo�A��FA��zA�a�Ao��@�BuA>��?6N<A+q�A��6A���A���A�V�A�A�r�A̝%AY�AI�@���A�aA���@%�C�ߌ@ZC�A-YA� �?�tAɄ�A΅BAm	�Ae�Aj�@���A���A�E0A�}�A�įA��@�eA|�|Ax$�AEWA�BrA�EC�h AІB6>A�yAQ�CA�z�@��]>�FTA�W�B�)A���A�}A��Al
@�I�A? g?D�A,�A�{�A�fA���A��3A��A�`�A�|�AY��AI	@�A�A��Z@$��C��@\,A�A��   
               (   5   #            e                                    )      "         "      .            	      
       +               	                  z   	   D   !      !            3   +   %            #                                    !      )   !      3      1   1      #            !   !                                  !      -                        )                                                                  +      '                                                                -   NG5�ON+�7NA�4NB-�O�mP�OwN��GN�e�N` JOȶO9Z�N�$N�BcN�;RN��Nվ�NZ�vNE*�O�)�N%�oN�:�N���N7��O1�_O� O�AP.��NQ/P�EO�bN)�N�d�O��N��SO��OR�O���O/�;NzǢO<%8N��	O�M��N9��O�AOU�N(��O
6lNYu�O�͓O���  �  +  �  �  �  �  �  ,  1  .      �  �  ,  !  �    2  �  �    �  L  j  �  �  u  �  �  }    O  i    �  -    h  �  �  �  T  �  n  �  [  O  s  �  w  	  ���o�D���o:�o:�o<ě�;ě�<�9X;ě�;ě�<T��=m�h<t�<t�<49X<49X<D��<D��<�C�<�t�<�t�<�o<�t�=��<���=\)<�<�9X<�/<�9X<�h=C�<ě�=��<�h=�w=+=#�
=��=�P=8Q�=��=��=49X=@�=@�=D��=aG�=ix�=�F=�+=�{=��jhrt�����tjjjjjjjjjjda``dgrtx�������tlgd��������������lrt��������tllllllll�����������������������
#(/1.$#
���5:BO[ht�����thOIA<65������������������������������������������������������������������������������������������������������������������������HHKT^aeea\TTMHHHHHHH��������������������
#/2<>>=<0/-#wz���������zwwwwwwww<64<>EHKUVWZafaUH<<<����������������������������������������1,,-,1:BN[^giie[NB51xxz�������}zxxxxxxxx=7<BO[bhhhhd[ODB====,+//<HLSPHA<6/,,,,,,"##0<>B?<0,#""""""""���������������������������������������������""���������������
�����#*/0/,#���)06:GI2)������������" �����������������������)/5855)����������������������������������������/159BBN[\cb][TNDB=5/	"+/38974/-"
		&)5BJKMGB5)	���
#////0/.$#
 ��"#08940+#24127<HU_adgea^UHB<2��������������������MMO[\hu~�������uh\ZM��������������������788BOUOOHB7777777777���������������������������� �������A=@BO[c[WOCBAAAAAAAA��������


	 �����dnp{��������{nddddddz|����������������{zyiffkms������������y�L�Y�e�j�m�e�Y�L�L�I�L�L�L�L�L�L�L�L�L�L�zÇÓàìñùûùõìàÓÇ�z�q�j�n�r�z����������ùõñùú�������������������ſy����������������y�s�m�y�y�y�y�y�y�y�y�G�T�V�Y�W�T�G�F�<�A�G�G�G�G�G�G�G�G�G�G�m�y���������}�y�m�`�T�G�F�G�H�Q�T�`�e�m���ʼԼ׼޼ۼѼļ������~�n�f�a�f��������"�&�/�3�1�/�'�"��	�����������������"����������������������������������������������	��	���������������������������/�<�F�D�<�8�/�#�"��#�%�/�/�/�/�/�/�/�/�B�O�V�[�c�c�[�X�O�B�6�+�)�'�)�+�6�=�B�B��������*�+�'�&�������ݻ���������������ݿѿοѿܿݿ���������������������¿ĿѿӿҿѿϿĿ�����������������������������������s�q�f�e�f�l�s�������āăĆčĚĚĚčā�|�|�~āāāāāāāā�a�m�z���������������z�m�d�a�Y�U�Y�a�a�aE�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E������������������������������������������������������#�$��������������������A�N�W�R�N�A�5�(�(�(�5�9�A�A�A�A�A�A�A�A�ʾ׾�����׾׾ʾǾ����������ʾʾʾ������������������������������������������������r�n�l�r�~���������������������ܹϹù����¹ùϹܹ�āčďĚĞĦĩĪĦĚčā�~�t�h�h�g�h�tā�u�v�uƁƃƁƄƆƁ�u�h�a�\�V�P�L�O�\�h�u��������5�B�a�d�d�[�N�5�)������������������������������������������������������<�I�d�wŇŏőŇ�{�b�<�0����
���#�<�y���������¿������������y�`�T�O�R�]�m�y�������ʼμʼż��������������������������M�Z�f�l�l�i�f�_�Z�M�M�B�C�M�M�M�M�M�M�M�����ܹٹֹ۹޹��������������нݽ������������ݽнǽĽ������ƽп�������)�*�(���������������/�;�R�S�H�D�;�/�"��	�����������	��"�/�)�5�B�N�[�h�t�y�~�t�g�N�5�(�����)�(�3�5�=�F�M�C�A�5�3�������
���(�������� �����ۼּҼּڼ�������s�������������������������w�s�l�f�g�h�sùþ��������ýùìàÓÒÓÛàâì÷ùù����	�������	������������𾌾��������������������������������������ܻ����� �����ֻܻܻܻܻܻܻܻܻܻܻܽ����������y�s�g�p�y�����������������������������������������������������������������������Ⱥ���������������������������D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D������������������������������������Ҽ��!�:�@�?�.���ּ˼������������zÇÓàùý����������ùìàÛÓÎÇ�{�z I ; | k U 1 9 F @ T ?  ' � K ? � D : = / c    3 =   ] F ? 7 l 5 @ 6 L L H F & <  E S 6 O [ / f / q n '    l  e  �  �  �  X  k  T  �  �  q    �  �  �  �  �    n  k    E  �  �  T  �  T  |  #  W  �  e  C  �  4  3  /  �  n  {  �  �  �  \    `  6  �  n  ,  �  �  3  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  �  �  �  �  �  �  �  �  �  �  z  Z  3  	  �  �  w  5  �   �      	        �    #    �  �  �  v  =  �  �    v   �  �             �  �  �  �  �  �  �  �  �  �  �  �  �    �  �  �  �  �  �  �  �  �  �  �  {  i  W  E  1       �   �  �  �  �  �  �  �  �  �  ~  u  m  d  [  T  N  I  C  >  8  2    	    �  �  �  �  5  _  s  �  �  }  `  .  �  �  6  �  3  �  �  �  �  �  �  �  �  �  �  w  f  N  &  �  �  F  �  %    6  9  (  +  a  �  �  �      +  %    �  �  8     �  p  d  1  '          �  �  �  �  �  �  �  �  �  �  �  �  �  �  .  +  (  $  !          �  �  �  �  �  �  �  �  |  ]  =  �  �  �  �  �  �    
  
    �  �  �  �  �  �  o  H    �  	k  
  
�  '  �     z  �  �      �  u  �  W  
�  	�    w  �  t  �  �  �  |  s  g  T  >  !     �  �  �  �  p  S  (  �  7  �  �  �  �  �  �  �  �  �    |  {  z  y  y  x  w  w  v  u  ,  )  &  #            �  �  �  �  �  �  �  �  �  �  |  !                  �  �  �  �  �  �  {  k  ^  P  B  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �        �  �  �  �  �  �  �  �  �  x  g  W  F  (     �   �  �  �  �  �  1  1  /  ,  $        �  �  �  �  y  7  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  l  H  !  �  �  �  f  o  f  �  |  o  a  P  ;  "    �  �  �  o  @  	  �    A  B    �  �  �  �  �  �  �  �  �    i  R  <  '    �  �  �  �  �  �  �  �  t  g  X  G  5       �  �  �  �  c  =    �  �  �  �    n  �  �  
  1  F  L  H  0  	  �  �  4  �  y  ,  �  1  K  b  c  d  e  g  h  i  i  k  m  i  c  L  /  �  �  ]   �  a  k  m  l  �  �  �  �  �  �  �  y  \  7  �  �  y  D    a  �  �  �  �  �  �  �  �  �  �  �  �  p  U  4    �  �  X  �  u  n  g  `  Y  M  B  6  +  !         �  �  �  �  �  �  �  �    �  �  �  �  �  n  I    �  �  U    �  �  q  c  >  i  �  �  �  �  �  �  �  �  �  x  k  f  �  �  �  �  �  �  �  �  +  f  w  z  o  ]  C     �  �  �  n  K  K  9     �  �  1          	  �            �  �  �  i  )  �  �  8  �    O  J  E  @  ;  6  1  ,  (  #            �  �  �  �  �  �  �  �  �  �        J  _  h  d  W  B    �  �  !  �   �      �  �  �  �  �  �  �  �  �  z  i  `  W  D  /    �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  F  �  �  !  �  #   �  -  &          �  �  �  �  �  �  �  o  N  ,    �  �  N  �  �  �      �  �  �  �  �  �  �  {  Q    �    ]  �  �  \  e  g  c  Y  K  :  )    �  �  �  E  �  �  =  �  G  �  g  �  �  �  m  V  >  &    �  �  �  �  w  W  )  �  �  9  �   �  u  y  }  �  �  �  �  �  �  �  �  �  �  �  S    �  �  /  �  �  �  �  �  �  z  n  a  Q  ?  +    �  �  �  �  �  x  V    T  R  M  D  :  /  "      �  �  �  f  6    h  �  {    �  �  �  �  �  �  k  T  >  )       �  �  �  �  �  �  �  p  R  n  [  I  7  &      �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �    m  [  >  "    �  �  �  �  �  [  Q  G  >  2  &          %  6  ,      �  �  �  �  -  O  <  #    �  �  �  [  )  �  �  x  =    �  �  �  L  �  �  s  o  k  f  b  ^  Z  T  N  H  C  =  7  &     �  �  �  g  @  �  �    (  :  O  ^  {  �  �  �  :  �    G  O  -  �  
9  �  w  g  W  K  A  7  *    
  �  �  �  �  �  t  I    �  �  {  	  	  �  �  �  �  �  �  �  �  �  R  �  �    �  �  �  �    �  �  �  {  n  Z  @  !  �  �  �  ^    �  �  .  �  D  �  g