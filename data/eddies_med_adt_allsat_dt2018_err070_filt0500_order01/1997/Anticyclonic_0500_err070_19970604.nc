CDF       
      obs    4   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type                     	amplitude                   comment       �Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM���   
add_offset               min       ?h�t�j~�   max       ?�$�/��      �  �   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       N"A   max       P�b       �  |   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       ����   max       =��
      �  L   effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>�\(�   max       @E�(�\            effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ?������    max       @vp             (<   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @*         max       @O�           h  0\   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @��        max       @��@          �  0�   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       �t�   max       >`A�      �  1�   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A�A   max       B,�]      �  2d   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A�z�   max       B,��      �  34   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       ?3�t   max       C�g�      �  4   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       ?G�L   max       C�gf      �  4�   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max          �      �  5�   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min             max          E      �  6t   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min             max          9      �  7D   
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       N"A   max       Pk=      �  8   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6��C-   
add_offset               min       ?�333333   max       ?� ѷX�      �  8�   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       ����   max       >O�      �  9�   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>�\(�   max       @E�G�z�        :�   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ?�p��
<    max       @vp             B�   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @*         max       @O�           h  J�   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @��        max       @��`          �  K,   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         C�   max         C�      �  K�   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6��C-   
add_offset               min       ?�䎊q�   max       ?��\��N<     @  L�      	                
         V      �         U   	   !                                     '   L      O   V      
         $      
      |   �      5   +      "      %      O�1VO.�N��NaxXP$��O嵳N{��O�.�N׽�P�b N�\�PH�<O|�!N-<P��O;"\P	o�OI.�N"AO̄N�O�{�N��_N�0�N�(O�r�PG�N�O`��P��`O|�O�q�PmOa�O>�N#�VNʔ�P
tOO�O3�N軷P2�P�V�N��!Pf�5O�F�N���O��N�"]O�T OA�NȒ���ͼ�o�e`B�#�
��o��o��o:�o;o;�o;�o<#�
<e`B<e`B<e`B<�o<�C�<�C�<�t�<���<��
<�1<�9X<�9X<�9X<���<���<�`B<�`B<�h<�h=t�=t�=t�=�P=��=��=�w='�='�=,1=8Q�=D��=P�`=T��=T��=T��=T��=aG�=e`B=��
=��
SPV[g����������tg`[S��������������������pmnt��������xtpppppp��������������������^`hmty����������zmh^ /<HU`\YRF>=/#)36A6)	/;HORRQPH;/& 		'"&)36BINLGB96)''''���)5HOHKI<"��������������������������������
';GC/$
���%" !")5BCGNQXVNB5-)%V[gtu}tgc[VVVVVVVVVVa]\ag�������������ga��������������������������#+44-
������������������������������������������

����4<IU]_UIF<4444444444	)5NV[NB5)	�������������������/((+/<HLUVUUMH</////��������������������eeaacemz���������zme������.NebTM>5�����������������������������������������������$5NgfB������grt{�������������tgghddhnz�����������znh������������������ !)**)(���	#/7<<91/%'#!
	����������������������������������������)5BN[]ZWNEB;5+��������


�������
#+,+'#
���������������������B>BOh|������ythKHGHBwv�����'=>6%����wTahmpxqmjdaUXTRTTTTT�������	!++&������������������������������������������������������+-*#������[Z[`gtw|ytlgf[[[[[[[)ABOYcgsoh[PB6)��������������������������

��������������������������������������������������ĦĳĿ������������������ĿĳĦĠěĠĦĦ�n�zÇÐÍÇ�~�z�n�e�a�^�a�e�n�n�n�n�n�n�������������������������������������������������������������������v�a�M�I�X�g���H�a�m���������������a�T�H�;�/�!��"�/�HÇÓÚÛØÓÒÇ�}ÂÂÃÇÇÇÇÇÇÇÇ�?�D�<�;�6�/�"��	���������������/�;�?�������	������	���������������������I�^�jŔŞř�{�b�0�
��������ľ������0�I����������������������������������������D{D�D�D�D�D�EEEEED�D�D�D�D�DnD]D\D{�ʾ׾����
����	�����ʾ��������¾ʾ����������������������������������������6�B�O�[�a�o�t�p�h�[�6�����������)�6�C�O�\�h�uƁƊƉƁ�u�h�\�O�C�9�6�*�6�:�C�5�B�J�>�<�F�:�5�)���������������(�5�;�>�G�T�`�m�m�k�`�T�G�;�.�"����"�.�;�/�<�H�T�H�?�<�/�)�+�/�/�/�/�/�/�/�/�/�/��������������������������������Żſ���Ƽ�������������������������������������������������������y�l�4�,�-�4�G�T�m������������������s�g�f�`�\�f�s���������������������������������������������������'�)�0�)���������������[�h�tāĚĦĭĲİĭĦģĚčā�t�f�\�Y�[������(�0�<�<�5� ��������˿ͿԿ�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E͹ܹ�����$�%��������Ϲù��ùƹֹ��p�|�w�~�y�a�;� ��	����������������;�p�	���"�*�/�7�;�6�/�"����	����	�	��(�5�A�E�J�W�Z�Z�N�����������������������������������������n�f�a�`�k�����:�F�S�_�g�k�d�_�S�F�-�!������!�-�:�f�s���������s�o�f�Z�X�M�A�>�A�K�M�d�f���������ؼ�������������������������������ĽǽĽý�������������z��������ƧƳ������5�6�)������ƸƚƊƁ�wƎƟƧ��#�/�<�H�U�a�h�b�a�U�H�7�/�#������(�4�A�Z�^�f�p�s�~�u�s�f�Z�M�I�A�7�4�'�(�����ʼԼּؼּӼѼʼ��������������������f���������������r�@�4����������'�@�f���4�L�R�N�:�'�������л������ֻٻۼŔŝŠšŠŔŇŀ�{�n�k�n�{ŀŇŔŔŔŔŔ�H�U�a�zÑÇ�l�<�/�����������������/�Hù����������������������÷àÜÕàìùÓàãìð÷òìàÓÇ��|ÁÇÊÓÓÓÓ�����������������������y�l�i�^�\�X�l�y���нݽ߽����ݽнƽĽýĽннннннкr���������ֺغغɺ������������~�l�e�c�r���(�A�J�N�X�f�Z�N�A�5�(���� ��
��s���������������s�n�l�g�j�j�s�s�s�s�s�s Y @ 7 > N M y M 6 7 2 V B m  Y P = \ 6  i 7  < . = j @ l a 2 - 1 3 c a d L [ / e ^ N g : # Y  : c j  <  3  �  i  �  :  �  r  �  �  �  �    _  h  �  �  �  2  4  �  	     �  6  '  �  `  �  B  c  -  �  �  O  E  �    �  i  �  �  -  �  9  Y  #  0  �  �  �  A�t���`B��o:�o<�<��
<o<�j<���=�Q�<e`B>:^5<�<�o=���<���=H�9=o<�j<�h<�j=H�9=\)=49X<���=Y�=ix�=C�=�+=�
==49X=�=���=��=@�=,1=P�`=���=�o=P�`=]/>,1>`A�=aG�=���=��=��=� �=m�h=�j=���=�9XB	�B4�B
=B"q�B ��BOYB� A�AB��BoB"�YBv�B�B	=;B
��B�B��B�.B��BY�B&��BO�B�?BդB�B �B��BK�BU�B��B
��BX�B9�BX�B��B �6B 1RB��BfrB$�{B"��B�B��A��eB2�B/�B!��B,�]B	e�B2�B��B�$B	G�B?�B
2B"B7B �fBK�B��A�z�B�aB��B"��B?�B@B	�'B;&B@ZBOMB�B�B�aB&�B?*B�B�RB�	B B`B�BV�B@BC�B
�KB�)BD�B@�B��B E�B BCBF<B@B$B�B"�|B��B��A��B	vB?�B"3PB,��B	o�B5{BL�BИA���A�v)A�;M@�A�F�A��oAɟ�A�'2A���A�m�@�h�C���AUJ�AHT_A�ڈB�&A�,AAc�EA�~-A�s�@�a�Ah�AD�UA�'�AՅ�A�N�A��C�g�?3�tA��A���A��A�,@|�gAA�>A�A!��B�A���A=�Q@�EF@�N@��A��A��HA�k4A��A��A*hH@�XA��A��-A�y:A��A�.@�A�&rA�}�A��A���A��CA�|�@�	C���AU �AG%eA׃�B;YA�T-Ab�`A�_�A�[�@�AgzyAC$�A�n�AՋqAނ&A� �C�gf?G�LA��FA�BA��A�X�@��AB�A�A!FB�A�L!A>�@��@�5�@��A�}�A��	AϞ�AʼA��A*@�HA���A�q�      
                
         W      �         V   
   !                                     '   M      P   V               %            |   �      5   +      #      %      	               -   %            =      3         %      +               %               )         E      #   '               +            /   ;      7         #      #                     %               /      !                                                      9         #               #            #         5               #      O ��N���N��NAO��N�ON{��OX��NɥPK0IN�\�O�תO|�!N-<Ozb�O;"\O�OI.�N"AO̄N�On��N��_N��jN�(O=e�O�`kN�O>��Pk=O|�OߵO�GOa�N�e�N#�VNʔ�O��`OO�O3�N軷O�<�O��N��!PT��Ou�N���N�f5N�"]O���OA�NȒ  a    0  W  �  �  o  �  
  �  �  �  =  w  
  �  �  *  R  �  �  �  "  ^  U    �  �  b  �  �  
S  	�  N  �  �  �  M  �  �  �      N    R  �  3  �  i  I  &����u�e`B�t�;��
<49X��o;��
;D��<���;�o=�-<e`B<e`B=D��<�o=C�<�C�<�t�<���<��
<���<�9X<�`B<�9X=o=C�<�`B=o=,1<�h=�w=L��=t�=��=��=��=<j='�='�=,1=�1>O�=P�`=aG�=e`B=T��=�C�=aG�=ix�=��
=��
V[\gt���������tog][V��������������������pmnt��������xtpppppp��������������������lhffj������������zql"#/1<CHJIH?</#)36A6) "/;HJNNNMKH;/*$ #')46BHMKFB86,)����)59>=?<)�����������������������������
!
�����%" !")5BCGNQXVNB5-)%V[gtu}tgc[VVVVVVVVVVtmjnt�������������tt�������������������������

�����������������������������������������

����4<IU]_UIF<4444444444)5BNOVTQNMB5) �������������������.-/2<FHPOH?<9/......��������������������hfhjoz����������zpmh������)5ENHA)�����������������������������������������������%5Z\<)������grt{�������������tggiefkz������������zpi�������� ���������� !)**)(���#/5;7/.#���������������������������������������� %)5BNSWSNHB5)"��������


�������
#+,+'#
���������������������IGHO[ht}�����th[VSRI�������������TahmpxqmjdaUXTRTTTTT������ ))�������������������������������������������������������������������[Z[`gtw|ytlgf[[[[[[[)7BOWbfrnh[OB6)��������������������������

��������������������������������������������������ĦĳĿ��������������ĿĳĨĦğĢĦĦĦĦ�n�zÇÐÍÇ�~�z�n�e�a�^�a�e�n�n�n�n�n�n�����������������������������������������s���������������������������~�j�[�Z�g�s�T�a�m�n�y�z�~�z�u�m�a�`�T�P�L�M�T�T�T�TÇÓÚÛØÓÒÇ�}ÂÂÃÇÇÇÇÇÇÇÇ�	��"�/�8�8�6�1�/�"��	���������������	�����	������	�����������������������#�<�U�b�{Ňŉł�{�b�I�#���������������#����������������������������������������D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D}D{D�D�D��ʾ׾����
����	�����ʾ��������¾ʾ����������������������������������������)�6�B�O�T�[�_�^�X�O�B�6�)������)�)�C�O�\�h�uƁƊƉƁ�u�h�\�O�C�9�6�*�6�:�C�����(�)�*�)�$������������������;�>�G�T�`�m�m�k�`�T�G�;�.�"����"�.�;�/�<�H�T�H�?�<�/�)�+�/�/�/�/�/�/�/�/�/�/��������������������������������Żſ���Ƽ����������������������������������������T�`�m�~�����~�z�y�m�`�T�G�;�2�4�;�G�N�T������������s�g�f�`�\�f�s���������������������
�������������������������������'�)�0�)���������������tāčĚĦĬĬĩĦĜĚčā�x�t�m�d�h�i�t�������(�5�5�/�(�������ڿٿ޿��E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E͹ܹ������!�"�������ܹϹǹ˹Ϲڹ��;�T�a�j�h�\�/���������������������#�;�	���"�*�/�7�;�6�/�"����	����	�	��(�5�A�I�V�X�W�N�(��������������������������������������������w�p�k�m�}���:�F�S�_�g�k�d�_�S�F�-�!������!�-�:�M�Z�f�s���������s�f�[�Z�N�M�M�M�M�M�M���������ؼ�������������������������������ĽǽĽý�������������z��������������������$�%�������ƳƠƒƞƧư����#�/�<�H�U�a�h�b�a�U�H�7�/�#������(�4�A�Z�^�f�p�s�~�u�s�f�Z�M�I�A�7�4�'�(�����ʼԼּؼּӼѼʼ��������������������f�r������������r�f�Y�@�2����'�4�@�f�������������ܻԻѻԻٻ����ŔŝŠšŠŔŇŀ�{�n�k�n�{ŀŇŔŔŔŔŔ�U�aÁÆÃ�i�<�/�����������������/�H�U�����������������������ùïìâáìù��Óàãìð÷òìàÓÇ��|ÁÇÊÓÓÓÓ�y�������������������y�s�m�m�l�l�l�x�y�y�нݽ߽����ݽнƽĽýĽннннннкr���������ɺֺֺɺ������������~�m�f�d�r���(�A�J�N�X�f�Z�N�A�5�(���� ��
��s���������������s�n�l�g�j�j�s�s�s�s�s�s g > 7 = I B y : 4 : 2 Y B m " Y  = \ 6  $ 7  < * 4 j < g a ( ) 1 ( c a a L [ / V K N f 1 # C  9 c j  g  �  �  H  R    �  �  �  �  �  �    _  �  �  1  �  2  4  �  �     �  6  �     `  �  S  c  �  %  �    E  �    �  i  �  �  w  �    �  #     �  �  �  A  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  *  2  8  >  G  R  [  `  `  X  J  5    �  �  �  �  �  r  .              �  �  �  �  �  �  �  �  �  �  �  �  �  �  0  +  %          �  �  �  �  �  �  w  ]  C  *    �  �  M  S  V  S  I  ;  #  	  �  �  �  h  7  �  �  �  C    �    Q  k  z  �  �  v  d  M  /  
  �  �  y  @  	  �  �  c  �  K  2  J  ]  t  �  �  �  �  �  �  �  �  �  �  �  �  |  O  �  �  o  f  ]  R  G  <  2  &        �  �  �  I    �  �  X    3  Y  q  �  �  x  m  a  T  =    �  �  �  ]  *  �  �  S  �    	    �  �  �  �  �  �  t  G    �  �  K  �  M  �  %  �    j  �  �  �  �  �  �  �  j  j  ^  -  �  �  Q  �  ^  [  �  �  |  t  m  e  \  Q  C  2      �  �  �    ]  :    �  �  �    �  �    �  8  o  �  �  4  �  �  �  S  �  �  
�  �  �  =  :  5  /  *  $        �  �  �  �  \  (  �  �  ?   �   {  w  q  k  e  _  Y  S  M  G  A  :  4  -  &              �  W  �  �  	X  	�  	�  
  
  
  	�  	�  	}  	*  �  _  �  �  �  s  D  �  �  �  �  �  �  �  �  �  y  c  M  7       �  �  �  �  s      3  W  _  o  �  �  �  �  �  �  �  �  w  E    �  P  �  *  *  '  !      �  �  �  �  �  �  [  ,  �  �  b  *   �   �  R  Q  P  O  M  K  H  E  @  6  ,  "      �  �  �  �  �  x  �  �  �  �  �  �  �  �  �    q  b  X  R  N  K  ;  )    �  �  �  �  �  �  �  �  �  �  �  �  x  j  \  O  B  5  (      *  i  �  �  �  �  �  �  t  a  Q  >    �  �  �  '  �  �  �  "    �  �  �  �  �  �  }  e  f  Z  H  3      �  �  �  �    &  A  N  W  ]  ^  W  J  5      �  �  �  >  �  �  [    U  R  O  M  J  F  @  :  4  /  &        �  �  �  �  �  �  �  �  �  �      �  �  �  �  ~  =  �  �  W    �  (  0    o  �  �  �  �  �  �  �  �  w  R  $  �  �  u  -  �  h  �    �  �  �  x  p  h  `  Q  ?  .    	  �  �  �  �  }  \  <      _  b  ]  Q  >  )    �  �  �  T    �    m  �    e  �    P  y  �  �  h  =    �  �  �  �  {    �  �  9  C  %  �  �  �  �  �    u  a  E  !  �  �  �  \    �  �  n  @      
&  
R  
I  
;  
&  

  	�  	�  	�  	Z  	  �  o    �  �  @  �  o  �  	_  	�  	�  	�  	�  	�  	~  	[  	,  �  �  c  �  �  �  \  �  �  �  �  N  <     �  �  �  }  O  &    �  �  �  �  k  8  �  �  �  U  �  �  �  �  �  �  �  �  �  �  �  u  d  P  <  (      �  �  �  �  �    v  m  b  W  L  A  3  $      �  �  �  �  �  �  �  �  �  �  x  U  -    �  �  |  P  %  �  �  �  Q    �  @    0  B  L  L  D  1    �  �  �  �  �  p  L    �  Y  �    �  u  \  C  +          �  �  �  �  f  (  �    C  {   �  �  �  �  �  �  �  t  _  Q  L  =  $    �  �  �  e     �   N  �  �  �  �  }  v  l  b  V  I  ;  +       �  �  �  W    �  N  @  �  z  �  �    �  �  y    �  %  �  
�  
   	  �  �  �  �  %  �    V  �  �  �  �  �    �    �  9  S    	�  �  8  N  @  2  $      �  �  �  �  �  �  |  i  U  L  E  ?  8  2  
         �  �  �  H    	  �  �  W  C    �  5  �  5  �    E  Q  <    �  �  �  _  %  �  �  x  A  �  C  �  �  �  �  �  �  a  7    �  �  z  J    �  �  �  y  L    �  �  }      �  �  �  �  �  �     ,  2  *    
  �  �  �    q  �   �  �  �  �  �  u  i  ]  Q  F  :  /  #       �   �   �   �   �   �  e  h  e  ]  Q  @  $    �  �    E  �  �  S  �  w  
  �  �  I  &    �  �  �  �  �  �  �  �  �  i  I  $  �  �  �  L  �  &    �  �  �  r  O  ,      �  �  �  �  �  �  �  �  �  �