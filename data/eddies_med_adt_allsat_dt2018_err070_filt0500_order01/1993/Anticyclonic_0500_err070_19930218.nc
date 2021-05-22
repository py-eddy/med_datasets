CDF       
      obs    ?   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type                     	amplitude                   comment       �Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM���   
add_offset               min       ?`bM���   max       ?�XbM��      �  �   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       N��   max       PX"K      �  �   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       ��   max       =�1      �  �   effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>ٙ����   max       @E�
=p��     	�   �   effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ?�z�G��   max       @vu��R     	�  *x   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @!         max       @P@           �  4P   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @�3        max       @��          �  4�   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       �ě�   max       >	7L      �  5�   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A�F�   max       B.X,      �  6�   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A�|�   max       B.ч      �  7�   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       @ M�   max       C��      �  8�   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       @4�   max       C���      �  9�   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max          c      �  :�   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min             max          =      �  ;�   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min             max          9      �  <�   
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       N��   max       P>�n      �  =�   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6��C-   
add_offset               min       ?�;dZ�	   max       ?�����>C      �  >�   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       ��   max       =�1      �  ?�   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>޸Q�   max       @E�
=p��     	�  @�   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ?��\(�   max       @vu��R     	�  Jx   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @!         max       @P@           �  TP   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @�3        max       @�u           �  T�   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         =�   max         =�      �  U�   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6��C-   
add_offset               min       ?��O�M   max       ?��	k��~     �  V�                     K            
   	   ,      E               E      ,      ;   )      3   )   
   /      .   +   4         c                        J                                 	      $   	            	NWf�N&�wOPN�2�OK��N��P7REOI*�O��HN>�O�$N�r:O�<4OƇPW�9N~�N�c�NdjN��]PaSN(�hP*N�5�P"yO��+O��xO�6O�TN;bPX"KO:S�O���Py�O�U�N�VLN�{dP��N}y�N���N���ON"��Nl�N7�+PB�mO�#N$"�N0N,��O��N:D�O/O��O,TNEO�N�SQO<m|Oi�1N���N�o,N�T�N�=�N#&����������ͼ�j��o�o��`B���
;o;��
;��
;��
;ě�;�`B<o<o<t�<#�
<49X<D��<T��<e`B<�C�<�C�<�t�<�j<ě�<ě�<ě�<���<���<�h<��=C�=C�=\)=t�=t�=t�=�P=�P=�P=�P=��=��=��=��=�w=�w=#�
=T��=T��=Y�=ix�=ix�=y�#=�7L=�C�=�C�=���=���=�1=�1������������������������������������������������������������" #%026::30#""""""����������������������������������������/+)/5B[s|���togNBB=/c_`cghu���������{tgc!#/<HUanz�����nUH/!�~������������������XW[]cgjtux{�����tg[X������

������������')((%�����������������������������/Hpz{sb/#
�����������������������)6>BOWVOLKB6)#�����������������������������������������������	 ��������ghkotz����thgggggggg)BNZeosqg[NB5#��������������lpwy~�������������ul/<HUnz��~nUD</#32<HUaz�������znUJ<3;Haz��������naSOLOH;�����$.1)#�����������������������!)5BN[g��������gB������
#(('#
��";DIMOMH;,"	#/<Mamsu~zaU</	�������
�������|y}�����������||||||[VOCB>BO[bhnph\[[[[[����),+11543)������

�������������%)�����������������������������"&)35>BLNQNHB95)%$#)56685)%%%%%%%%%%����


�����������xz~�������}zxxxxxxxx%#1BU]V[��{}���|gB+%&" #()59BCJNPUNDB5)&|z��������||||||||||^amz{~zomlaa^^^^^^^^556BOSQOCBA655555555#(/1347:<=</)# fhkmuz��}zpmffffffff��������NOPV[^ht���ztshb[SON-)+/0;HTUY^_[TH;7/--:;<CHPURMHA<::::::::%)15BNRPNEB5)>=?BFHNTaflmnnmkaTH>���������������)66;:64)_Yadmrz�����zwma____����&'�����������������������������������������������������������������������������������Ŀ������������ĿĴľĿĿĿĿĿĿĿĿĿĿ�����������������������������������}�~���нݽ����������ݽнͽͽннннн������	��"�%�-�1�/�*�"��	�������������𿸿��ĿƿĿ��������������������������������6�t�|�}�w�h�[�6�)�
����������������a�n�zÇÒÓàçäàÖÓÇ�z�n�k�a�\�[�a�����������������������������z�����������������������������������������������\�h�n�uƁƍƎƑƎƁ�u�k�h�\�O�B�G�M�U�\��������������������������������čĚĳĸ����������ĿĳĦĚčā�z�y�zāč�������ʼݼ�޼ͼ������������x�t�����������$�3�Q�O�A�7�(�����꿾�ǿ�����m�y���������~�y�m�e�`�X�`�e�m�m�m�m�m�mÓààáåìñøìàÛÓÇ��{ÃÇÎÓÓ��"�+�/�/�/�&�"�������������z�������������~�z�m�k�g�m�n�z�z�z�z�z�z��5�N�Y�_�]�N�5�(���������������;�>�G�T�Y�T�N�G�;�9�4�:�;�;�;�;�;�;�;�;��=�I�Z�b�S�=�0����������������������ŭŹ����������������������ŽŹŭŬŬŭŭ���4�M�Z�_�c�Z�H�4���	�	�����ӽ�����������
���������������������E�E�FFFFFFFE�E�E�E�E�E�E�E�E�E�E��	���������������������������������	�;�H�T�g�k�g�`�T�H�/������������	��/�;ÓÓØßÓÇ�}�{ÇÑÓÓÓÓÓÓÓÓÓÓ�)�t���p�P�L�R�v�w�l�g�d�B�5�(�����)����)�D�N�X�T�N�K�B�5�)�"��������	��"�/�H�T�_�i�u�m�a�H�;�/� ��	��� �	�A�Z�����������m�f�Z�M�A�=�K�K�0�%�+�4�A���$����������ܻлɻ��������ûӼ���������ĿĿ̿Ŀ����������������������������������������������������������������/�H�T�g�z���������z�m�a�T�����������/�������������s�f�d�\�f�s�y�������`�`�d�f�i�h�`�S�H�J�M�Q�S�_�`�`�`�`�`�`�O�S�O�M�O�O�C�8�6�*�*�)�*�4�0�6�C�D�O�O�"�.�;�G�L�T�U�T�R�G�C�;�7�.�%�"����"�����������߾ݾ����������D{D�D�D�D�D�D{DoDmDmDoDuD{D{D{D{D{D{D{D{�ĿƿѿӿѿǿĿ��������ÿĿĿĿĿĿĿĿĿ���(�=�N�����������s�N�5���ݿο˿ݿ������������������������������������������'�4�:�=�4�4�2�'��!�'�'�'�'�'�'�'�'�'�'�\�^�d�^�\�O�K�C�B�C�O�P�\�\�\�\�\�\�\�\�����
���������������������������	��"�:�;�=�;�.�"��	�����ھ�����	ŇŔŠŤŭŵŭŠŔŏŇńŇŇŇŇŇŇŇŇ�ʾ׾����������߾׾Ծʾ��������þʾʻ-�0�:�F�Y�_�k�k�_�S�F�:�/�-�"�!��!�(�-�0�<�I�T�U�Z�_�Z�U�I�<�0�#��� �#�-�0�0EEEEE#EEED�D�D�D�EEEEEEEE�<�I�L�U�Y�^�[�U�I�F�<�8�4�8�<�<�<�<�<�<�h�tāčĚĞĦĪĪĦĚĒčā�{�t�`�\�\�h�r�~�����������������~�s�e�Y�L�I�N�Y�e�r�r�y�~���������~�r�f�e�Y�e�i�r�r�r�r�r�rìù����������ÿùìàÚßàæëìììì�!�-�:�;�F�G�F�:�-�#�!���������!�Y�f�r�����������������r�q�f�[�Y�R�Y�Y�����ʼּؼ�ּʼļ��������������������� I X = F ) e < 2 W n I 9 3 , Q ! W K \ $ > , < A [ [ B P < } H > F O U ] X X � f C X Q 2 p : 9 3 1 f h  J  9 ^ < 3 < V H U q    h  +  _  �  �  O  6  �  	  [  W  $  �  �  "  z  9  �  �  d  Q  �    �  �  �  /  ?  h  F  �  �  �  �  �  �     y  $    7  B  �  L  �  J  O  W  K  �  �    b  k  U  �  �  �  �  �  '    l��j�ě��e`B�u;D�����
=�7L<�C�<���<t�<u<e`B=L��=\)=���<49X<��
<u<�o=��T<u=m�h=�P=���=q��=P�`=��P=�o=+=�hs=#�
=��P=��=�{=#�
='�>	7L=,1=0 �=#�
=L��=#�
=H�9=,1=�`B=Y�=<j=,1=8Q�=m�h=ix�=�O�=��T=��=�7L=�O�=��=���=���=� �=�j=���=�v�B"3�B~�B0BB%�2B�B��B`JB
%�B͠B �B	��B>:B��B!RtB�'B,�DB��B�B�B4BP�B�=B��B �B�VB��BN�B��B!��B	8�B�A�F�B,HB��B>]B�B��B#ǰB.X,B/�B4�B�BZB I�BhB^8BA�~�BH�B<A���BƍB�-A�LfBA>BPA��RB�rBLrA�"pBB)B{�B"==B[�B&�B%�bB!�B�CB<B
 �B��B ��B	��B=�BB�B!ôB�
B,�jBĹB<�B?rB;�B@*B��BA�B A&B�MB@�BK�BL�B"=�BA�B��A�|�BLBDB*�B7�B�B#�B.чB��B=�B�8B?�B ASB@>B?�BݡA�8BA%B�A�w]B�hB��A��B?�B�QA�vpB�eBB�B :�B�cB1UB@@�lA�M>A�d�A,�A�P�AvY�A؃fA��4A�Q�B+gB?2A���A��%@�=�A�g[Ak�A�!�A��rA��A��AdB�B�A��@A78�A�YC��A�f A���A��A�� A���A��AAx@�F�Au�A�}A�b�AD"�A�2B ��Ab�AW<C���AxX�A�gA�@�!�Bw?@�EA[ۚA�~AS��@�VUA�C�`�A�_LAݥ�@�t@ M�A��@l��@��D@��
@	{A��A��A,�A�v&Av�Aؑ�AȀ|A�k�B��BD�A��*Aߊ,@��?A��PAl�YA�E7A�}@A�#A���Ac�B�A�iA8_A�{�C���A�}�A�ؒA�=�A��~A���A���AAh�@��Au��A��7A�uAC��A$B F5Ab�AWs�C���Ax�A�-A�r�@�?B~�@�iA\��A�zAS��@���A� C�fBA��wA�1@�1@4�Á�@pu@�=o@���            	         K               
   -      E               F      -      ;   *      3   )   
   /      .   +   4         c                        K      	                           	      %   	            
                     -      #            !      5               %      )      +         %   '      =         /   !         +                        7                                                                                 #                  #                     %                        9                                                                                                   NWf�N&�wNZ+mNm):NՆZN��O8LO"O��HN>�O�$N�r:O ��OrV�O�D(N~�N�c�NdjN��]OM�N(�hP�#N���ONqnN��O��xO�EzO���N;bP>�nO:S�O��=OU�hOt-JN�VLN�{dOeR$N}y�N���N���ON"��Nl�N7�+O��O�#N$"�N0N,��O��N:D�O/O�[O,TNEO�N�SQO<m|O��NO�kN�o,N��JN�=�N#&�  K  �  /  �  k  {  X  �    J    u  I  *    h  /  �  s  	t  1  '  R    c  )  U  �  �  {  #    p  	  �    �        �  �  Z  �  	  ^  �  T  S  �  <  �  �  l    )  �  F  �  `  }  }  ͼ��������
��9X�t��o<��%   ;o;��
;��
;��
<�`B<u<�<o<t�<#�
<49X=49X<T��<���<���=<j=t�<�j=��=o<ě�<�h<���=+=H�9=,1=C�=\)=��P=t�=t�=�P=�P=�P=�P=��=�\)=��=��=�w=�w=#�
=T��=T��=]/=ix�=ix�=y�#=�7L=���=�\)=���=��-=�1=�1������������������������������������������������������������#"#+0159810#######����������������������������������������58>BN[egnqog[XNB:655ddghrt|��������utogd!#/<HUanz�����nUH/!�~������������������XW[]cgjtux{�����tg[X������

�������������	
 ���������������������������#/<HUbnqrofH</#��������������������)6>BOWVOLKB6)#������������������������������������������������������������ghkotz����thgggggggg$)5BNU_log[NB(�����������������������������������/$$)/<HOUZUTHA</////32<HUaz�������znUJ<3aZZY\anz���������zna������#&*&�����������������������)5BNg��������gB������
#(('#
��"0;AFJLMKG;'"#/<>EHKMI>/#���������������|y}�����������||||||[VOCB>BO[bhnph\[[[[[����()+*)%�������

�������������%)�����������������������������"&)35>BLNQNHB95)%$#)56685)%%%%%%%%%%����


�����������xz~�������}zxxxxxxxx4115<BN[gnqqpme[NB>4&" #()59BCJNPUNDB5)&|z��������||||||||||^amz{~zomlaa^^^^^^^^556BOSQOCBA655555555#(/1347:<=</)# fhkmuz��}zpmffffffff��������WOQVZ[_ht����ztrh^[W-)+/0;HTUY^_[TH;7/--:;<CHPURMHA<::::::::%)15BNRPNEB5)>=?BFHNTaflmnnmkaTH>��������������������%!")16986.)%%%%%%%%_Yadmrz�����zwma____����$������������������������������������������������������������������������������������Ŀ������������ĿĴľĿĿĿĿĿĿĿĿĿĿ�����������������������������������������нݽ�����������ݽнϽϽннннн������	��"�#�#�"���	�����������������𿸿��ĿƿĿ������������������������������C�O�[�_�d�b�[�Z�O�B�6�2�+�)�� �)�6�>�C�n�zÄÇÓÕàãàßÓÇ�z�n�g�a�`�a�f�n�����������������������������z�����������������������������������������������\�h�n�uƁƍƎƑƎƁ�u�k�h�\�O�B�G�M�U�\��������������������������������čĚĦĬĳĹĺĶĳĦĚđčćąĊčččč�������ʼμؼռϼʼ�������������������������1�B�D�B�=�5�)�������ڿڿ���m�y���������~�y�m�e�`�X�`�e�m�m�m�m�m�mÓààáåìñøìàÛÓÇ��{ÃÇÎÓÓ��"�+�/�/�/�&�"�������������z�������������~�z�m�k�g�m�n�z�z�z�z�z�z��(�5�A�F�N�O�P�I�A�;�5�(��������;�>�G�T�Y�T�N�G�;�9�4�:�;�;�;�;�;�;�;�;������$�0�I�P�X�U�I�=�0���������������Ź��������������������ŹŭŭŹŹŹŹŹŹ����4�A�F�L�P�P�M�A�4�(������ �����������	�������������������������E�E�FFFFFFFE�E�E�E�E�E�E�E�E�E�E������������������������������������������;�H�T�X�b�e�b�Z�T�H�;�/��	��� ���1�;ÓÓØßÓÇ�}�{ÇÑÓÓÓÓÓÓÓÓÓÓ�)�5�]�y�z�i�M�O�i�q�t�g�[�B�7�+��	��)����)�D�N�X�T�N�K�B�5�)�"���������/�H�T�Z�f�m�s�m�a�T�H�;�/�"��	����M�Z�f�s�����������w�s�f�Z�M�B�?�A�H�M���������������ܻлû����ûлܻ鿟�������ĿĿ̿Ŀ����������������������������������������������������������������;�H�L�T�_�h�e�T�H�;�/� ������"�/�;�������������s�f�d�\�f�s�y�������`�`�d�f�i�h�`�S�H�J�M�Q�S�_�`�`�`�`�`�`�O�S�O�M�O�O�C�8�6�*�*�)�*�4�0�6�C�D�O�O�"�.�;�G�L�T�U�T�R�G�C�;�7�.�%�"����"�����������߾ݾ����������D{D�D�D�D�D�D{DoDmDmDoDuD{D{D{D{D{D{D{D{�ĿƿѿӿѿǿĿ��������ÿĿĿĿĿĿĿĿĿ������#�,�/�)�������ݿٿؿڿݿ������������������������������������������'�4�:�=�4�4�2�'��!�'�'�'�'�'�'�'�'�'�'�\�^�d�^�\�O�K�C�B�C�O�P�\�\�\�\�\�\�\�\�����
���������������������������	��"�:�;�=�;�.�"��	�����ھ�����	ŇŔŠŤŭŵŭŠŔŏŇńŇŇŇŇŇŇŇŇ�ʾ׾����������߾׾Ծʾ��������þʾʻ!�-�:�F�S�X�_�j�k�_�^�S�F�:�0�-�#�!��!�0�<�I�T�U�Z�_�Z�U�I�<�0�#��� �#�-�0�0EEEEE#EEED�D�D�D�EEEEEEEE�<�I�L�U�Y�^�[�U�I�F�<�8�4�8�<�<�<�<�<�<�h�tāčĚĞĦĪĪĦĚĒčā�{�t�`�\�\�h�e�r�~���������������~�r�e�e�Y�V�Y�\�e�e�e�r�~���������~�r�k�e�`�e�e�e�e�e�e�e�eìù����������ÿùìàÚßàæëìììì�!�-�7�:�F�:�9�-�!�!���������!�!�Y�f�r�����������������r�q�f�[�Y�R�Y�Y�����ʼּؼ�ּʼļ��������������������� I X L E - e ! 6 W n I 9 ( ? ( ! W K \  > , 6 0 - [   K < v H ; . H U ] 0 X � f C X Q 2  : 9 3 1 f h  H  9 ^ < 6 0 V E U q    h  +  f  �  �  O  }  A  	  [  W  $  $  �  �  z  9  �  �  �  Q  z  �  �    �      h  �  �  e  �  �  �  �  �  y  $    7  B  �  L  �  J  O  W  K  �  �    I  k  U  �  �  1  c  �  �    l  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  K  I  H  E  >  7  +    
  �  �  �  �  �  �  �  �    u  k  �  �  �  �    y  t  o  j  d  ]  R  H  >  4  *           �  �  �  �  �    #  -  .  -  %      �  �  �  �  x  z  �  �  �  �  �  �  �  �  �  s  `  L  6    �  �  �  a  &   �   �  �    7  O  ^  f  k  k  c  Q  ;       �  �  �  f  4     �  {  p  f  \  R  H  >  3  (         �   �   �   �   �   �   �   �  �    ^  �  (  �  �    @  O  V  A    �  x    y  �  �  �  �  �  �  �  �  �  �  �  �  �  r  S  2    �  �  ^    �  S          �  �  �  �  �  �  �  w  N    �  �  6  �  �  �  J  <  -        �  �  �  �  �  �  �  �  {  g  Q  <  &      
    �  �  �  �  �  �  �  �  �  x  b  U  L  9  $    �  u  l  c  Z  R  D  4    �  �  �  �  �  n  V  =  !     �   �  �  �  �         *  :  B  I  H  @  .    �  @  �  9  y  �  �  �  	    "  )  &          �  �  �  r  C  �  �  k    +  d  p  z  �         �  �  z  ,  �  ]  �  M  �  C  4   �  h  a  Z  S  L  E  >  5  *      	   �   �   �   �   �   �   �   �  /  )      �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  }  s  k  e  _  Y  U  U  U  V  M  >  .      �  �  �  s  f  X  J  <  +    	  �  �  �  �  �  �  �  �  �  l  U  ?  �  e  �  ,  �  �  	  	E  	h  	s  	`  	  �  R  �  3  h  d  M  �  1  +  &  !               �   �   �   �   �   �   �   �   u   e  �    %  "    �  �  �  H           �  �  W    �  �  >  (  ?  R  Q  L  B  1    �  �  �  �  o  F    �  �  R  �  �  �  +  K  g  �  �  �  �        �  �  �  m    �  �  �  �  u  �    k  �  �  '  J  _  `  S  +  �  �  w    �  �  x    )  �  �  �  z  >  ;  >  /      �  �  �  �  \  �  +  e   �  �  �  %  @  L  S  T  I  /    �  �  ]    �  s    n  �  �  5  j  �  �  �  �  �  �  ~  s  O    �  �  u  %  �  F  �  �  �  �  �  �  �  �  �  �  {  l  ^  O  ?  /           �  �  6  s  y  h  S  <  %    �  �  �  u  F    �  1  G  �    k  #    
  �  �  �  �  �  �  k  B    �  �  {  B    �  q   �          �  �  �  A  �  �  �  e  7    �  k  �  �    !  �      @  S  Z  b  h  o  i  V  ;    �  �  �  >  �  &    �  �  �  	  �  �  �  �  h  @    �  �  7  �  &  y  �  o  U  �  �  �  �    q  d  \  W  R  H  :  ,      �  �  �  �  �    
    �  �  �  �  �  �  �  �  �  �  |  Y  7     �   �   }  	�  
j  �    O  �  �  �  �  �  I  �  �    
}  	�  �  �  �  �      �  �  �  �  �  �  �  �  �  �  �  [    �  �  _  7      �  �  �  �  �  �  �  �  �  �  �  p  ^  L  ;  )          �  �  �  �  �  �  �  �  �  �  �  �  ~  m  \  K  :  )    �  �  �  �  �  �  �  �  n  Z  D  ,    �  �  �  Q    �  _  �  �  �  �  �  ~  u  j  _  T  I  =  2  #    �  �  �  �  �  Z  X  U  O  A  1    	  �  �  �  �  S  #  �  �  �  [    �  �  �  �  �  �  w  j  ]  O  B  0      �  �  �  �  �  �  o  O  7  .  �  �  �  �  	  	  	  	  �  `  �  Y  �  Z  �  �  �  ^  U  H  8  '       �  �  �  �  z  U  %  �  �  t  <     �  �  �  �  �          �  �  �  �  �  �  �  �  �  n  6  �  T  N  I  C  =  7  1  (        �  �  �  �  �  �  �  �  �  S  O  J  F  A  =  8  0  '          �  �  �  �  �  �  �  �  �  �  �  m  R  /    �  �  �  Z  1    �  �  �  I  c  �  <  0  %      �  �  �  �  �  �  �  �  �  �  �  �  �  k  U  �  �  �  �  n  W  ?  &  
  �  �  �  t  G    �  �  =  �  y  �  �  �  �  w  Z  7    �  �  x  ;  �  �  q  #  �  �  �  -  l  l  Z  A  +      	  �  �  �  �  �  �  x  A  �  �  t  4       �  �  �  �  ^  7    �  �  �  ]  *  �  �  �  I  
   �  )        �  �  �  �  �  �  i  K  -    �  �  �  �  x  p  �  �  Y  0    �  �  �  Y  (  �  �  �  >  �  �    r  �  8  �      .  <  D  E  ?  /    �  �  m     �  y     �  =  �  �  �  �  �  �  �  �  �  �  �  �  �  �  k  C    �  �  m  /  `  C  $  �  �  �  t  _  M  ?  @  Q  a  q  h  U  A  -      z  }  n  S  1  
  �  �  �  [  +  �  �  �  Q    �  Z  �  �  }  x  p  b  O  9  "    �  �  �  �  e  >         �  �  �  �  �  �  �  �  �  �  p  W  :    �  �  �  �  T    �  �  Q