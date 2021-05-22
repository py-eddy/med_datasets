CDF       
      obs    7   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type                     	amplitude                   comment       �Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM���   
add_offset               min       ?h�t�j~�   max       ?�V�t�      �  �   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       N��   max       P��^      �  �   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       ��w   max       =��m      �  d   effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>�        max       @E������     �   @   effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ��\(�    max       @vh��
=p     �  (�   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @)         max       @Q`           p  1p   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @�9        max       @�v@          �  1�   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       �+   max       >���      �  2�   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A���   max       B-f	      �  3�   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A��O   max       B,ۜ      �  4t   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       =H�   max       C�	�      �  5P   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       =d�F   max       C��      �  6,   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max         ]      �  7   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min             max          C      �  7�   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min             max          9      �  8�   
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       N��   max       P�z�      �  9�   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6��C-   
add_offset               min       ?�_��Ft   max       ?�:)�y��      �  :x   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       ��w   max       >333      �  ;T   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>�G�z�   max       @E������     �  <0   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ���Q�     max       @vh          �  D�   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @#         max       @Q`           p  M`   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @�9        max       @���          �  M�   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         ?�   max         ?�      �  N�   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6��C-   
add_offset               min       ?���@��   max       ?�64�K     0  O�        ]            *         $         `                     Q               0      &                      )            	   +      [   )      0   u         �   c            7      N�UxOiM�P��^N���N��O�O��N��O�8�O��O>F�O�P�CO�0�O��OH�TO5J�OS)<OT�P�)�N��eP��Nw1lNO|3Oȯ~N��O�sN��NW�P �O*1OR�^OL:O�juN�lZO/=�O��gN��P��O?��P�OTdNeUO���P][N��O�!O�	3O��8OL�N���Nzb%Ol�
N�@jNP���w��t���o�D���D��;�o<o<t�<D��<D��<T��<u<�C�<�C�<�C�<�t�<���<��
<��
<�1<�1<���<���<�/<�`B<�`B<�h<�h<�=+=+=+=�w=�w='�=,1=8Q�=<j=@�=H�9=L��=Y�=aG�=ix�=q��=q��=y�#=�o=�o=�7L=�hs=�t�=��==��m��������������������#',0<IUX]][ULI<0#')7g����������tgNB+'ttx����������ttttttt��������������������1;<?N[gtu}~wg[NGB;51>J[t���������tg[NIC>@=BBN[^[NLBB@@@@@@@@�������������������������
����������������������������������������������� 5t���������gB)�)5BNUZ]][UNB))58BBEBA:5)$��������������������	"/;?BA;6/"	������

�����#)/5BNUNDB=5)�������8A6����������������������������
#/<Fb^UH?/#
;<IU\^ZULIFE@<;;;;;;����������������yvuvz�������������y��������������������������)5652)���WST[`ghtw}}{{zthc[WW������������������������5BM@B@C@5��LLLNV[gt����vtig[NL���������mlmoz�����������zsom��������� ������BHUVafnz��zxnlaURHBB��������������������&%)06O[hrxtqh[OKB6/&�����������������������);BOX\\Q@6���������

���lfinz�������������vl��������������������������������������zsu}���������������z&(6BOhlt|�}}wp[B6-)&aghtx|ytihaaaaaaaaaa���)..+)%!�����������
  
�������������������������
##&&%#
���VV]amyz{zxmaVVVVVVVVwroqz������zwwwwwwww�����
#$)+)$#
����QNOTUamz~�zuma_TQQQQ!!"#,/<;4/)#!!!!!!!!ĳĿ��������ĿľĳĳĮĦģĦĦĬĳĳĳĳ�������������������������x�m�_�[�`�o������)�B�r�w�j�a�B�6�)����ìÔÌÑî������������¹ùɹ̹ù������������������������������%�&������������������������"�/�;�H�U�V�U�J�H�/�"�����	� ���"���������������������������������������ſ"�.�;�;�D�=�;�7�0�.�"��"�"�"�"�"�"�"�"���������������������������������������ž�(�A�M�e�n�����s�Z�M�(���������������������û̻лܻܻлȻ����������������a�m�s�z�������������z�m�k�a�^�W�R�T�[�a¦¢�t�g�N�5������"�5�ʾ׾�����
�� ���׾ʾ��������������h�uƁƌƎƐƐƎƃƁ�u�j�h�\�S�T�\�d�h�h�'�4�C�M�Y�f��~�r�f�b�Y�L�4�'��
���'�T�a�d�m�u�z�������z�m�a�N�H�C�D�H�I�S�T�<�H�U�a�e�f�a�U�Q�H�<�1�#�����#�/�<�Z�f�j�q�s�v���������z�f�c�Z�T�O�Q�W�Z�"�/�H�Z�j�r�r�T���������������������"�A�N�R�Z�[�]�g�s�v�s�g�Z�N�I�A�@�9�7�A�A�	�/�;�a�k�n�l�e�Y�H�"�	���������������	�����������������������������������������T�`�m�o�r�p�m�m�h�`�\�V�T�S�T�T�T�T�T�T�5�A�N�Z�g�{��y�s�h�Z�N�������(�5�����������������������������������������������
��� �����������������������!�-�:�B�F�S�T�U�S�F�:�-�!������!�!���Ǽʼ̼˼ʼ���������������������������������(�.�A�I�S�N�A�5�(����տؿ���Z�g�s�������������������s�g�e�Z�X�N�R�Z�x�������������x�l�S�-�!�� �-�:�F�S�l�x����������������������ƵƧƣƚƘƚƧƳ�������������������������ƳƪƥƩƳƽ����� ���������������������������������������������ĽͽннŽĽ������������׾���	������	���ɾ��������ľʾ׾4�A�M�Q�S�T�O�M�A�>�4�-�(�.�4�4�4�4�4�4�r�~�������úֺ̺�ƺ��������r�h�`�`�h�r�Z�f�s���������������s�f�Z�M�A�A�G�M�Z������������ܹù������������Ϲ��a�a�g�n�r�zÅÇÓàìï÷ôàÓÇ�z�n�a�.�;�G�H�G�F�;�.�&�+�.�.�.�.�.�.�.�.�.�.�zÇÓì�����������������ùìÓÌ�z�u�z���1�L�^�n�Y�P�@�4����ܻ˻����ûͻ鼘���������������������������������������#�0�<�B�I�K�P�R�I�<�0�,�#������#�#D{D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�DyDtDsD{�����������ʼټ߼߼ܼʼ����������������������������ǺǺ�������������������������ŭŹ����������ŹŰŭŤŤŭŭŭŭŭŭŭŭ�F�S�_�l�x�l�k�_�S�F�E�B�F�F�F�F�F�F�F�FE�E�E�E�E�E�E�E�E�E�E�EuEiEhEgEkErEuEwE�āčĚĥĦĮİīĦĚđčĄāā�yāāāā���
�����
�������������������������� M  0 S 5 + 6 w * 6 2 E Q 8   g ) @ e * x = 7 l @ > " 1 O G J � *  l  % I B 0 < 1 6 X E / 1 / c  ' E * = :    �  �  �  �      �  `  B  *  �  W    p    �  �  �  e  *    �  �  �  �  �  >    U  ~  �  p  �  �  &  s  �    e  �  �  �  !  %  �  +  <  �  �  9  �  �  �    ^�+��`B>���;D��;D��<�C�=H�9<49X<�/=@�=��<�/=�l�='�<�`B=t�=0 �=�w<�`B=��`<�`B=H�9<�`B=o=���=t�=�+=0 �=C�=P�`=49X=T��=�\)=���=D��=�+=�o=]/=�Q�=�\)>\)=�v�=y�#=���>333=��=��
>s�F>&�y=�-=��w=��->�u>V>O�B_/B&`tB	lrB�BރB��B	X"B3�B��B#`�B"��B�TB	�B�B�3B"�A���B%LB�(B�$B	B�B&��B-f	B�BiB�B�B#.�B�eB	$�B�B HB��BK�B!h�BȡBAuB$[B��B�B!�+B�B �B�RB	|B�zBCB>�B$o�A�	B\KB�*A���B6�B
ۆB&|�B	B�B>�B +B��B	@>BŊBj�B#u�B"E3BSRB
:�B��B�uB"��A��OB;BVB6oBtB�`B&�sB,ۜB ��BP�BåB�6B#=�B9B	��B��B @oB��B�B!�kB��B�FB9�BY�B��B">�B;BC�B��B=�B�B=�BCEB$F�A�_�B~�B��A�\B@A��@���A�=H�A���A���A�"Aa�A���A8��@���A��A�FAT7RB��@�cA��^A�J�A@JA���A�G-A��@��Aij.A��PAs�A��[@w�@�6�A�F�A��J@���B��B�jA�HEA#�AV �A:�H@�LAA�'>�h6A�.�Ab�3A��@���@���A��C��7@��k@�3A�}8@��eC�	�A�,}A�qA�s@���Aԗ�=d�FA���A��A��^A`�5A�|�A7k�@�A�}�A��HAT�.B�9@͸�A��A�ktA@�A��A�{�A�]:@���Ai yA���As�xA�~~@{E�@�
�A��$A��X@�?gB��B�8A��`A#�AT�A;4�@|AB�U>��UA�ggAa�fA���@�~�@��A��C��@���@�)A��L@���C��AߟA狹        ]            *         $         a                  	   R               0   	   &                      )            	   ,      [   )      0   u         �   d            8               C                     '         ;                     A      +                        +      !      !         !      '      %         "   +            %                                                !         )                     9      %                              !               !      #                  !                              N�UxOiM�O��~N0�hN��O4rN�YgN��O�8�O���O4�O�P+pvO?�O��O1�N���OS)<Nd4�P�z�N��eO��Nw1lNO|3Oq��N��O�sN��NW�OƜzOT�OR�^O�<O�j`N�lZO/=�O��gN��O�O5�O��O-�NeUO���O�aN��N�O\�Ow&OL�N���Nzb%O<swN�@jNP�  1  v  "  k    �  �  <  9  ]  (    �  �  S     �  8    `  �  �  �  R  �    &  �  �  e  c  �     �    �  &  ]  �  �  
�  %    �  	  P  �  A  '  N  �  �  ?    Ž�w��t�>333�o�D��;�`B<��<t�<D��<�t�<e`B<u=8Q�<ě�<�C�<���<�`B<��
<�j=+<�1<�`B<���<�/='�<�`B<�h<�h<�=t�=C�=+=49X='�='�=,1=8Q�=<j=L��=L��=���=y�#=aG�=��=Ƨ�=q��=�%=�l�=ě�=�7L=�hs=�t�=��`==��m��������������������#',0<IUX]][ULI<0#GDEHP[gt�������tg[NGy|����������yyyyyyyy��������������������CBDEN[gmstxxtmg[WNCCSSZ[gt|���~xtga[SSSS@=BBN[^[NLBB@@@@@@@@��������������������������
��������������������������������������������3BNg���������[5))5;BELNLDB5))58BBEBA:5)$��������������������")/2;=;;4/"������

�����)5;85)������%6;4)���������������������������	#/<CU^]YQ<#	;<IU\^ZULIFE@<;;;;;;����������������|{~�����������������������������������������)5652)���WST[`ghtw}}{{zthc[WW������������������������)5;<@=5���OONPY[gt��~tthg[OOOO���������vroosz�����������zvv���������
	�������BHUVafnz��zxnlaURHBB��������������������&%)06O[hrxtqh[OKB6/&�����������������������6BOUZZO>6��������

����yz����������������}y��������������������������������������}|�����������������}0/06BO[hksutroh[OB60aghtx|ytihaaaaaaaaaa �)),*)$ 	  �������

�������������	 ������������
##&&%#
���VV]amyz{zxmaVVVVVVVVwroqz������zwwwwwwww�����
"&)&#
����QNOTUamz~�zuma_TQQQQ!!"#,/<;4/)#!!!!!!!!ĳĿ��������ĿľĳĳĮĦģĦĦĬĳĳĳĳ�������������������������x�m�_�[�`�o��������)�6�@�G�G�B�6�)���������������������ùùŹù��������������������������������%�&������������������������"�/�;�H�P�Q�M�H�;�:�/�"�������"�"�����������������������������������������"�.�;�;�D�=�;�7�0�.�"��"�"�"�"�"�"�"�"���������������������������������������ž�(�4�A�M�^�g�v�u�Z�M�A�(���������������������û˻лڻлĻ������������������a�m�s�z�������������z�m�k�a�^�W�R�T�[�a�[�t�t�g�[�B�3�+�)�.�0�8�N�[�׾���� ���������׾ʾ��������ʾ;��h�uƁƌƎƐƐƎƃƁ�u�j�h�\�S�T�\�d�h�h�'�4�A�M�Y�r�x�r�f�`�Y�I�4�'������'�T�a�l�m�x�z�{�z�q�m�a�a�T�N�N�S�T�T�T�T�<�H�U�a�e�f�a�U�Q�H�<�1�#�����#�/�<�Z�f�m�o�o�f�Z�X�S�T�Z�Z�Z�Z�Z�Z�Z�Z�Z�Z�"�/�;�V�h�i�T�/�	���������������������"�A�N�R�Z�[�]�g�s�v�s�g�Z�N�I�A�@�9�7�A�A�	�/�;�T�a�i�h�b�V�H�0�"��	�����������	�����������������������������������������T�`�m�o�r�p�m�m�h�`�\�V�T�S�T�T�T�T�T�T�5�A�N�Z�g�j�n�i�]�Z�N�;�5�(�$����'�5�����������������������������������������������
��� �����������������������!�-�:�B�F�S�T�U�S�F�:�-�!������!�!���Ǽʼ̼˼ʼ���������������������������������&�+�5�<�=�=�5�(����޿�����Z�g�s���������������s�g�f�Z�Z�P�Z�Z�Z�Z�x�������������x�l�S�-�!�� �-�:�F�S�l�xƧƳ��������������������ƺƳƩƧƝƠƧƧ����������
�����������ƺƳƬƦƫƳƿ����� ���������������������������������������������ĽͽннŽĽ������������׾���	������	���ɾ��������ľʾ׾4�A�M�Q�S�T�O�M�A�>�4�-�(�.�4�4�4�4�4�4�~�����������Ѻֺú��������r�j�a�`�b�k�~�f�s���������������s�f�Z�M�E�C�H�M�Z�f������������ܹϹù����������ùϹܹ�ÇÓàèìððìàÝÓÇ�z�v�o�q�z�}ÇÇ�.�;�G�H�G�F�;�.�&�+�.�.�.�.�.�.�.�.�.�.ÇÓìù������������������ùìßÓÈÀÇ������7�D�G�C�@�4�'������ܻջϻػ鼘���������������������������������������#�0�<�@�I�J�N�I�I�<�0�.�#������#�#D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D~D�D����������ʼѼ׼ּѼʼ������������������������������ǺǺ�������������������������ŭŹ����������ŹŰŭŤŤŭŭŭŭŭŭŭŭ�F�S�_�l�x�l�k�_�S�F�E�B�F�F�F�F�F�F�F�FE�E�E�E�E�E�E�E�E�E�E�E�EuEoEjEnEuEvE�E�āčĚĥĦĮİīĦĚđčĄāā�yāāāā���
�����
�������������������������� M   U 5  % w * / / E H &   Z , @ Z 3 x > 7 l , > " 1 O 0 C � (  l  % I > . )  6 Y 9 / ) , ?  ' E ( = :    �  �  r  p    Q  �  `  B  �  ~  W  )  �    �  �  �  �  V    +  �  �  �  �  >    U  �  G  p  S  �  &  s  �    1  �  v  +  !  �  �  +    �    9  �  �  �    ^  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  1  6  ;  ?  :  3  ,  $        .  B  T  `  l  w  z  }  �  v  n  g  b  \  V  P  I  ?  3  '        �  �  �  �  �  �  �  �  e  �  �    :    �  �      �  �  �     �  s  ]  ,  )  -  5  B  U  j  l  �  �  �  �  [  U  �  �  �  �  �  �  H    �  �  �  �  �  �  �  �  �  �  �  �  �  �  �      ~  ~  z  u  o  l  {  z  p  d  X  K  >  /      �  �  �  �  x  N  �    r  �  �  �  �  �  �  �  �  �  �  �  z  2  �  `  �  �  <  -         �  �  �  �  �  �  �  m  S  9       �   �   �  9  6  0  )  #        �  �  �  �  �  �  e  D  (  	  �  �  %  F  V  ]  U  =    �  �  �    \  -  �  �  E  �  g  �  S    (  $    �  �  �  �  Q    �  �  a    �  v  ,  �  k  '      �  �  �  �  �  �  k  B    �  �  x  0  �  �  Z     �  Q     O  �  �  �  �  �  w  @    �  �  u  F  �    ?  [  �  y  �  ~  y  t  z  �  t  [  8    �  �  �  K  �  �  T     �  S  N  J  G  G  G  I  K  E  <  .         �  �  �  f  �  7  �  �  �  �  �  �  �  �  �  �  u  ]  E  -      3  R  e  o    >  Y  n  }  �  �  �  �    g  I    �  �  _    �  R  V  8  (    �  �  ~  �  �  u  j  b  _  S  >     �  �  �  2  �  �  �  �  �  �  �  �      �  �  �  �  �  �  �  �  �  �  �  )  G  W  _  N  7    �  �  b  4      �  �  G  �  �  �  �  �  �  �  �  �  �  �  �  �  �  {  d  A    �  �  �  Q     �  �  �  �  �  �  �  �  �  i  J  ,        �  �  �  #  �  P  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  R  L  G  B  <  6  0  *       �  �  �  �  �  �  �  �  �  �  �  (  ^  �  �  �  �  �  y  [  4    �  o    f  �  �  �  �        �  �  �  �  �  �  �  �  i  Q  5    �  �  �  k  #  &    �  �  �  �  a  )  �  �  �  o  X  N  )  �  d  �  P  �  �  �  �  �  �  �  �  �  �  �  ~  j  U  (  �  �  �  =  �  �  �  �  �  �  �  �  �  �  �  w  l  c  Z  Q  I  D  @  <  8  4  '  ]  d  e  c  ^  U  J  =  +    �  �  �  �  U  
  �  [   �  T  ]  `  S  E  7  &      �  �  �  �  �  �  �  {  _  B  &  �  �  �  �  �  �  �  `  ;    �  �  �  �  �  �  �  �  �  �  �  �  �     �  �  �  �  �  �  z  F    �  t    m  �    Z  �  �  �  �  �  x  b  I  (    �  �  �  R    �  4  �  �  �        �  �  �  �  �  �  �  �  �  l  N  +    �  �  �  V  �  �  �  �  �  �  r  Y  9    �  �  o  7    �  �  F  �  �  &  "      �  �  �  �  �  |  X  .  �  �  �  �  �  T   �   y  ]  R  H  <  /  "      �  �  �  �  �  �  �  y  c  B    �  v  �  �  x  c  @  
  �  |  %  �  �  F  "  �  �  0  �  T  (  �  �  �  �  �  �  u  W  0    �  �  W    �  �  L  �  ~  �  
K  
y  
�  
�  
�  
�  
�  
�  
�  
�  
F  	�  	j  �  ;  �  �  �  9  �  �  �    #  %      �  �  �  �  S    �  ]  �  F  �  �            �  �  �  �  �  �  �  �  �  �  �  �    
      �  �  �  �  �  �  y  M      �  �  �  v    �    u  �  �  <    W  �  �  �      �  �  `  �  p  �  
  
  �  �  �  �  P  =  )       �  �  �  �  �  u  _  J  8  /  %        �  �  �  �  �  �  �  �  �    \  2    �  �  s  :  �  �  C  [    �  �    .  <  9     �  �  T  �    '  �  g  �  t  #  	"  	  �  �  �  �    '    �  �  }  !  �  4  
�  	�  �  |  V  �  N  ;  *      �  �  �  �  q  L    �  �  }  >  �  �  )  �  �  s  ^  K  <  ,    �  �  �  �  �  �  q  \  J  7  3  4  5  �  �  �  �  �  �  �  �  �  �  �  |  q  e  Z  O  L  L  L  L  +  1  >  ?  8  !    
�  
�  
X  	�  	�  	,  �  E  �  Q  �  g  �          �  �  �  �  t  I    �  �  e  &  �  �  b  F  ]  �  �  �  �  �  y  e  \  =  �  �  �  :  �  �  ^    �  q  �