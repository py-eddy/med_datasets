CDF       
      obs    A   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type                     	amplitude                   comment       �Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM���   
add_offset               min       ?`bM���   max       ?��+J       �   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       Njw   max       PuTR       �   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       �\)   max       =Ƨ�       �   effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>�z�G�   max       @E�          
(   �   effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ����
=p    max       @vE��R     
(  *�   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @&         max       @P�           �  5   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @�e        max       @�ـ           5�   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       ��`B   max       >���       6�   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A���   max       B,D�       7�   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A�yB   max       B,?�       8�   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       >�)�   max       C���       9�   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       >��N   max       C��	       :�   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max          �       ;�   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min             max          9       <�   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min             max          1       =�   
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       Njw   max       P)�       >�   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6��C-   
add_offset               min       ?�=�K]�   max       ?�_ح��V       ?�   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       ���   max       >hs       @�   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>�p��
>   max       @E�          
(  A�   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ����P    max       @vBz�G�     
(  K�   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @&         max       @P�           �  V   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @�e        max       @��           V�   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         =�   max         =�       W�   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6��C-   
add_offset               min       ?}��Z�   max       ?�_ح��V     P  X�                                 	   
      
   ?                        4      
      	         6   ^         #   
   *   7   @            F      B      /            
      :                     V   �   	   	          N�DN��NjwO�LNW��OthO1�@Ow�O���N3EHN��AN���N�1�N]�PuTROsq�OL>(O�VN�zO���O^��N��P\<Oޝ(N�*+Nu�NX]�Og��O?�P$P4��Nk�N�^nO:d�N�(IO��]OˊtO�Q,N'N�#�O\.P��N�j�O�/ZOi;�O��5NI�O��EO�J�Nb�O���P*vHNN_\OEOO
�O�Od=�N`��O�tO��Nf�iN��O4��O��N��\)���ě��u�e`B�49X�49X�t�%   ;o;��
;��
;�`B<o<T��<T��<T��<T��<e`B<u<u<u<�C�<�C�<�C�<�t�<���<�1<�9X<�9X<ě�<ě�<���<���<���<���<���<�/<�/<�`B<�`B<�`B<�h<�h<�<��<��=+=t�=t�=�P=�P=�w=#�
=49X=H�9=H�9=Y�=]/=y�#=�%=�O�=�{=�{=Ƨ�jgiijrtu�������tjjjj� ������������������������������xuz����������������x(),6BFOSOB;64)((((((BB<6346<BOQVY[\[WOLB��������������������##*0<IOX``RIC<830)'#��������������������!)1)).5BFB?;54)!JNNW[glrng[NJJJJJJJJ�������������������������������������������)1<?:AQUNB)���#!!$)5BNZ[cih[NB50)#,-/2<UahnunaYUQH<1/,)6BOXhqsuqh[B6,��������������������������������AOP[ht|�������th[YLA����������������������	"HammcTH;/"	��)B[gjmligZNB5$Z[gghtu{�����tjgca^Z����������������������������������������.*'&*./<HU_da[UPH</.��������������������������
#**&��������)5BHLB>5��� #0770*#          ���������������������������������llmyz|���������zrmll����������������������������������������ACKV`gtz�������tgNBA��������������������������
�������������������

������#/<HTXYYZWQH<���� ������!#.6BO[ht}|}vh\O6)!�����
#++(%!
�����
#Uaz���zna</#
��.,003<IIID<0........��������������������lihz��������������zl��������������������"%#%/;>HOTXYYXTHF/$"33535;Ng������tg[N;3WW[htyyth[WWWWWWWWWW*&#&/<CFHLUadg`UH</*��������������������UX\admoz}�����~zmaUU��������������������('')*67;?<860)((((((�����6BIKHA6)������������
 
���]afhtu����th]]]]]]]]20./69BOSJCDHJIB=762����� ����1469=BBEO[^choh[OB61���������������������U�a�n�zÇÓÕÓÊÇ�z�n�a�U�K�I�U�U�U�Uìù��������ùìæåìììììììììì����������������������������������������ĳĿ��������������������ĿĳĭĬĪĦīĳÇËÓÞÞÓÈÇÅ�~�z�y�zÅÇÇÇÇÇÇ�`�`�l�x�������������������������y�m�l�`�����	�� ����	�������������������������������ݽнĽ��������Ľнսݽ��������4�>�4�'�����ܻû��ûлܻ��Z�f�q�j�f�`�Z�V�R�X�Z�Z�Z�Z�Z�Z�Z�Z�Z�Z�)�-�6�A�B�D�F�B�>�6�)�����'�)�)�)�)�O�[�\�h�l�m�h�[�O�I�D�K�O�O�O�O�O�O�O�O��(�5�?�A�N�V�P�N�A�5�(�#��������T�a�k�i�b�a�^�T�I�K�T�T�T�T�T�T�T�T�T�T��0�I�{ŔŧŠŇ�n�<����������������
����ʾ׾����������׾ʾ����������������	���"�'�(�&�"��	���������������� �	�����������¾������s�f�Z�M�?�A�L�^�f�����y�{���������������y�v�s�y�y�y�y�y�y�y�y��������������������������������������������
����
���	���������������������#�/�8�<�H�P�H�G�<�<�/�'�#��� �#�#�#�#�;�H�a�e�i�g�Y�T�H�;�"��������,�/�9�;������������ ������ƳƩƧƳƹƻ������ƎƏƚƝƚƏƎƁ�{�u�h�\�V�U�\�`�h�uƁƎ�{ŇŐňŇ�{�n�g�n�n�{�{�{�{�{�{�{�{�{�{���������������������������������������Ҿ���	��"�;�<�A�?�8�.�"��	����������*�1�6�>�C�D�L�C�6�*���
�	������)�5�B�F�H�[�g�g�^�N�)��������������)�����������������������������q�s�����żּ��������ۼּԼּּּּּּּּּֿ	��"�.�;�G�K�H�G�;�.�"���	���	�	�	�	àìù����������ùìàÓÇÆ�ÁÇÉÔà�s�����������������v�s�g�Z�Z�Z�\�g�q�s�s���������������������������������|�������ùϹܹ�����'�/�'����ܹù����������ÿѿݿ����5�>�5�/����ݿѿ¿��������ĿѼ��������������������������������������������ĿͿ˿Ŀ����������������������������ѿݿ����	���������ݿۿտѿͿοѿ����(�Z�s�������������N��������������������������������������������������лܼ��+�4�9�;�<�4�����ۻлƻ������оM�Z�f�s����������s�f�Z�M�A�7�(�#�4�A�M������'�/������������������������������������������v�|���������G�T�`�g�y�}�{�y�m�]�T�G�;�.�)�#�#�%�<�GĦĳĿ��������������ĿĳĦĝĒďďĒĚĦ�y�����������y�t�l�`�]�`�l�v�y�y�y�y�y�y������#�:�@�<�:�#����������������������������������������������s�a�U�W�c�u�����ѿݿ���ݿѿʿɿͿѿѿѿѿѿѿѿѿѿѾ(�4�A�M�d�g�Z�M�=�4�.�*�(������!�(�T�a�j�m�e�a�\�T�M�H�;�9�;�;�8�:�;�H�M�TŔŠŭŴŹ��������ŽŹŭŠŜŔŐōŊŔŔ�H�U�a�n�zÁÀ�z�n�a�U�H�<�4�8�<�F�K�C�H���������������������������������������׼������ּ�ּҼʼ����������}�s�q�s�����DoD{D�D�D�D�D�D�D�D�D�D�D�D�D�D{DmDfDgDo�F�S�_�a�`�_�U�S�F�@�@�B�F�F�F�F�F�F�F�F���!�-�7�:�>�=�:�-�!�������������r�~���������������������~�r�e�Z�W�Y�b�r�����������ɺʺɺǺ���������������������������(�)�(� ����������� I L H O C I > ] X h M 6 H : [ ' . E f $ d 7 ? , ^ H V 1 ' A # > c - 0 N Y C W J B c = O L h 0  ] I T F ? l | 4 J ] * ) W � $ : Q    -  >  -  �  �  ^  �  x  K  b  �  �  �  �  �  �  �    V    �  �  �  �  #  .  �  �  7      >  �  �  �  �  @  ;  Z  �  �  '  �  R  �    :    W  y  �  ?  u  �  �  ?  �  �    #  �  G  �  9  Q��j��`B��C�;ě��o;D��;ě�;��
<�9X;ě�<e`B<e`B<49X<�C�=���=+<�/=�P<�o='�<�1<�/=�O�=t�<���<��
<�/='�<�`B=���=��<�/<�h=u=\)=�7L=��T=�^5=+=o=e`B=Ƨ�=49X=��=]/=��-=t�=<j=e`B=<j=ix�=��=49X=P�`=T��=�%=�%=ix�>\)>���=�hs=��-==��`=��B	��B��B"�aB��B��Bm B�FB&wYB!�hB*�B	9B�YB/B�BB�B~�B�wB�B"�B)�B �A���B�B	�YB��Bz�B�NB7!B�)B��B%z�BU�B!ԦB �B6�B|B	z[B"7_B�<Bb�B�xB�B��B$K�B�OB&(�B ��BlB,D�A���B	8�B��B�B��A���BB��B�.B�B!`Bb�B�_B��B� B
'$B�B"ILB�hB�(B|�BbcB&@PB!ÔBP�BB�[B=�B�KB@0B�BR�B@BB��B?�B��B�>A�yBB��B	��BPJB�BX�B7[B��B��B%{�B=PB">�A���B@�B=[B	��B"?�B��B?�B� B�1BCqB$?)B7dB&;�B Q_B�B,?�A��gB	�"B��B��B�A��YBJ�BȗB��B>�B7>B��BA�B�TB,ZA���A�i�@q�A���A���AsA�V�A*�S@��A@�A�A�(kA�y�A�<A�VsARA���AD�pAn�<A�A��mA���A�:�B	`B�XA�A�خA]+�A�R�A�rA�o]A�A_įA���A���A�Bh>�)�A~��@���Av�/A�|A���@��@��A?D	A�t@�D�Af5LA�rA?DA���A�`A|�A9*�A�SdA��AŃ�A���@��C���@�? @d�4@�@m�A3��A�p�A�#@��A�A� A%�A�~�A+�@��A?A֊2A��A� �A�nQA韇AS�A���AH�dAn�A��nA�?A��:A�y�BE�B|CA��}A�SMA_ A��NA���A�u�AڊA_�AˌgA��^A�[�>��NA~�d@��.Av��A&A���@��@��A?� AҀ1@�bAhu�A�}0A�A�dHA�<�A|ߵA:��A�R�A�F2A�	EA���@��zC��	@��@kE�@�@"��A3<'                                 
   
      
   @                        5      
      	         7   _         $   
   *   8   A            F      B      /                  :         	            W   �   	   	   !                                                   9         '               %   #                  +   )                  %   %            '      %      '                  /                     #   !                                                            1         '               !   #                                                                                    #                                       N�!N��NjwO���NW��OthN�c-O	 OO�N3EHN��AN���N�1�N]�P)�Osq�O0YO�VN�zOO�&O^��NnXBO�gnOޝ(N��~Nu�NX]�Og��O?�O-qOQ��Nk�N�^nO;(N�(IOfF0Ou�QOC�N'N�#�O(۶OF�;N�j�O�krO'%O�4�NI�O��EO�J�Nb�OQzPa^NN_\OEOO
�O�Od=�N`��O���O^/�Nf�iN��O,�aN�;N�  -  �  �  �  w  J  �  �  �  �  '  (    5  "    �  �  
  )  �    t  R  �  ~  v  K  Y  #  	�  �  �  V  y  s    �      �  	  m  �  p  �  ?  �  S    u  �    �  g  �  r  \  �  n  L    3  �  y�����ě��T���e`B�49X�ě��o;�`B;o;��
;��
;�`B<o<���<T��<�o<T��<e`B<���<u<�t�<ě�<�C�<���<�t�<���<�1<�9X=L��=��P<ě�<���<�<���=\)=#�
=Y�<�/<�`B=o=q��<�h=,1=\)=8Q�<��=+=t�=t�=49X=0 �=�w=#�
=49X=H�9=H�9=Y�=�O�>hs=�%=�O�=� �=� �=Ƨ�onmpt|�����toooooooo� ������������������������������}y�����������������}(),6BFOSOB;64)((((((BB<6346<BOQVY[\[WOLB��������������������*(%0<IMUV^]UPIE<:40*��������������������!)1)).5BFB?;54)!JNNW[glrng[NJJJJJJJJ��������������������������������������������)3782<LKB5)�#!!$)5BNZ[cih[NB50)#801<AHU[aimda]UNH<88)6BOXhqsuqh[B6,��������������������������������AOP[ht|�������th[YLA��������������������";Tafh_QH;/"	)B[gjmligZNB5$fdbgtx�����togffffff����������������������������������������.*'&*./<HU_da[UPH</.��������������������������

��������� "'**)(�� #0770*#          ���������������������������������llmyz|���������zrmll����������������������������������������SPRV[]gtv�������tg[S��������������������������
���������������������������$(/<?HKMNMKH<7/*&%#$���� ������))+.26=BO[htulf^RB6)�������
$#!
��!!#/<HUn|~znaUH</$!.,003<IIID<0........��������������������lihz��������������zl��������������������,/1;HHNQTTPH;331//,,>7:BN[gt�������tg^N>WW[htyyth[WWWWWWWWWW*&#&/<CFHLUadg`UH</*��������������������UX\admoz}�����~zmaUU��������������������('')*67;?<860)((((((�����6;BEA;1)��������

 �����]afhtu����th]]]]]]]]20./69BOSJCDHJIB=762������������86:=BO[\bhmhf[OB8888���������������������a�n�zÇÍÇÅ�z�n�a�W�Y�a�a�a�a�a�a�a�aìù��������ùìæåìììììììììì����������������������������������������ĳĿ��������������������ĿĳįĮĬĨįĳÇËÓÞÞÓÈÇÅ�~�z�y�zÅÇÇÇÇÇÇ�`�`�l�x�������������������������y�m�l�`����	�����
�	�����������������������ݽ��������޽ݽнĽ����������Ľн׽ݻ������!�'�)�'�������ܻܻ���Z�f�q�j�f�`�Z�V�R�X�Z�Z�Z�Z�Z�Z�Z�Z�Z�Z�)�-�6�A�B�D�F�B�>�6�)�����'�)�)�)�)�O�[�\�h�l�m�h�[�O�I�D�K�O�O�O�O�O�O�O�O��(�5�?�A�N�V�P�N�A�5�(�#��������T�a�k�i�b�a�^�T�I�K�T�T�T�T�T�T�T�T�T�T��#�0�I�{ŏŖŇ�{�n�I�������������������ʾ׾����������׾ʾ������������������	��"�"�#�$�"���	�����������������������������¾������s�f�Z�M�?�A�L�^�f�����y�{���������������y�v�s�y�y�y�y�y�y�y�y��������������������������������������������
����
���	���������������������#�/�<�H�K�H�C�<�7�/�*�#�#�"�#�#�#�#�#�#�;�H�T�a�e�c�X�R�M�H�;�"�	� �����/�6�;������������ ������ƳƩƧƳƹƻ�������h�uƁƋƉƁ�w�u�h�\�Z�X�\�f�h�h�h�h�h�h�{ŇŐňŇ�{�n�g�n�n�{�{�{�{�{�{�{�{�{�{���������������������������������������Ҿ���	��"�;�<�A�?�8�.�"��	����������*�1�6�>�C�D�L�C�6�*���
�	������)�5�?�B�N�P�V�Q�N�B�5�)����
���%�)�����������������������������������������ּ��������ۼּԼּּּּּּּּּֿ	��"�.�;�G�K�H�G�;�.�"���	���	�	�	�	àìùÿ����ÿùìàÓËÇÂÄÇÐÓÝà�s�����������������v�s�g�Z�Z�Z�\�g�q�s�s�����������������������������������������ùϹܹ߹����� �����Ϲù����������ÿѿݿ��� ����������ݿѿ̿ĿſʿѼ��������������������������������������������ĿͿ˿Ŀ����������������������������ݿ��������������޿ݿؿӿѿѿ׿��g�s������v�g�Z�N�A�5�*���$�5�A�N�Y�g�����������������������������������������лܻ������#�-�3�4�'�����ܻлǻŻо4�A�M�Z�f�s�x������s�f�Z�M�A�>�4�1�/�4��������������������������������������������������v�|���������G�T�`�g�y�}�{�y�m�]�T�G�;�.�)�#�#�%�<�GĦĳĿ��������������ĿĳĦĝĒďďĒĚĦ�y�����������y�t�l�`�]�`�l�v�y�y�y�y�y�y���#�-�%�#��
�����������������
������������������������������s�i�Z�]�g�|���ѿݿ���ݿѿʿɿͿѿѿѿѿѿѿѿѿѿѾ(�4�A�M�d�g�Z�M�=�4�.�*�(������!�(�T�a�j�m�e�a�\�T�M�H�;�9�;�;�8�:�;�H�M�TŔŠŭŴŹ��������ŽŹŭŠŜŔŐōŊŔŔ�H�U�a�n�zÁÀ�z�n�a�U�H�<�4�8�<�F�K�C�H���������������������������������������׼��������ʼͼͼʼ¼�����������y�v�x���D�D�D�D�D�D�D�D�D�D�D�D�D�D�D{DyDxD{D~D��F�S�_�a�`�_�U�S�F�@�@�B�F�F�F�F�F�F�F�F���!�-�7�:�>�=�:�-�!�������������e�r�~�����������������~�r�e�[�Y�X�Y�c�e���������ɺƺ���������������������������������(�)�(� ����������� H L H L C I * \ : h M 6 H : j ' 5 E f  d B ; , N H V 1 ' + & > c * 0 D :  W J 5 X = P F ` 0  ] I [ : ? l | 4 J ] * # W � $ 5 Q    �  >  -  �  �  ^  �  <  [  b  �  �  �  �  �  �  /    V  �  �  }    �  �  .  �  �  7  j  �  >  �  C  �  �  �  �  Z  �  t  �  �  k  o  L  :    W  y  Z  �  u  �  �  ?  �  �  v  �  �  G  q    Q  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�       $  (  *  ,  ,  ,  *  %         "  )  )  &       �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �      '  6  F  �  �  �  �  �  �  �  u  b  J  1    �  �  �  `     �  �  U  �  �  �  �  �  �  �  x  a  J  3       �  �  �  �  b  �  q  w  r  m  h  `  W  N  C  6  *          �  �  �  �  �  �  J  C  >  7  '    �  �  �  �  �  �  n  R  7    �  �  �  E  �  �  �  �  �  �  �  �  �  �  �  �  t  W  8    �  �  ]  �  �  �  �  �  �  �  �  �  �  �  �  n  V  Y  O  (  �  �  9   �  ^  j  p  q  m  �  �  �  n  Y  B  %    �  �  v  )  �  ^  i  �  �  �  ~  p  b  T  F  8  *       �  �  �  �    `  A  "  '    
        �  �  �  �  w  U  0    �  w    �  w  2  (      �  �  �  �  �  �  �  n  [  G  3      �  �  �  �      �  �  �  �  �  �  �  �  �  s  W  <        �   �   �   �  5  5  6  )      �  �  �  �  �  �  �  �  y  b  =    �  �  l  �  �    "    �  �  �  �  B  �  L  <  �  �  8  �  �  r        �  �  �  �  �  �  �  ~  l  T  7    �  �  Z  �  O  �  �  �  �  �  �  �  �  �  �  �  �  �  �  t  P    �  7   �  �  �  �  �  �  �  n  X  <       �  �  �  t  -  �  �   �   k  
  	                    �   �   �   �   �   �   �   �   �   ~  	    &  )  &      �  �  �  �  �  �  j  A    �  C  �    �  �  �    t  j  a  W  N  F  ?  <  8  6  7  8  8  3  /  +  �  �  �  �      �  �  �  �  �  �  �  t  [  A  &    �  �  +  d  r  s  j  R  .    �  �  �  �  K  
  �  X  �    �    R  K  9  "    �  �  �  �  o  W  A  (    �  �  �  �  ?   �  �  �  �  �  �  �  �  �  �  �  �  �  m  N  ,  
  �  �  �  x  ~  }  {  y  x  v  t  s  q  o  n  n  m  l  l  k  k  j  i  i  v  v  w  q  g  ]  P  B  1      �  �  �  �  �  �  t  \  C  K  C  A  6  )      �  �  �  �  �  �  v  X  .  �  �  6  \  Y  T  O  K  ?  1  #      �  �  �  �  �    f  O  D  8  -  �  �  �  �  �  �  �  �      #    �  �  }  "  �  �  K   G  �  �  .  �  	  	W  	�  	�  	�  	�  	�  	�  	�  	<  �  0  m  V  �    �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  |  j  Y  E  /       �   �   �   �   �   �  /  @  Q  U  I  6    �  �  m  #  �  �  ?  �  �  9  �  I  A  y  q  i  [  M  <  )      �  �  �  �  �  l  H    �  �  �    <  ^  o  s  p  e  Q  4  
  �  �  7  �  }    �  �    :  �  �  �  �  �          �  �  �  e    �    {  �  �  H  �  L  �  �  g  �  �  �  �  �  �  �  D  �  p  �  <  �  �  �    w  o  h  a  Z  T  L  C  :  2  *  #        �  �  �  �      �  �  �  �  �  �  �  p  a  S  E  7  *     �   �   �   �  �  �  �  �  �  �  �  �  �    ]  6    �  z  )  �  h  �  s  ,  �  �  #  T    �  �  �  �  �  �  �  8  �  F  �  6  �  �  m  _  N  ;  &    �  �  �  �  l  R  /    �  �  c  '  �  �  e  �  �  �  �  �  �  �  �  �  x  ?  �  �  4  �  
  _  �  �  2  U  c  l  p  m  b  L  4    �  �  �  �  _    �  _  �  �  �  i  �  �  �  �  �  �  �  �  �  �  �  f  ?  �  h  �  �  l  ?  :  5  0  +  $          �  �  �  �  �  �  �  �  �  j  �  �  �  �  �  �  v  `  J  4       �  �  �  �  �  �  d  4  S  O  =  (       �  �  �  �  �  v  L    �  �  :  �  �  `    �  �  �  �  �  �  �  �  �  �  v  [  ;    �  �  �  �  Z  �  �    /  N  d  r  u  q  h  Z  I  4      �  �  �  z  X  j  �  �  �  �  �  �  �  �  x  \  2  
  �  �  J  �      0        �  �  �  �  �  �  �  �  �  �  �  �  s  \  C  +    �  �  �  �  �  �  �  q  U  7    �  �  �  �  �  �  �  �  �  g  V  D  0      �  �  �  �  {  l  \  B  '        $  1  �  �  �  �  �  o  ]  H  #  �  �  �  S    �  �  �  |  ^    r  a  N  :  '    �  �  �  �    X  -    �  �  l  C    �  \  O  A  4  '      �  �  �  �  �  �  �  �  �  �  {  m  _  �  �  �  �  �  �  t  N    
�  
�  
0  	�  	#  �  �  �  �  P    �  M  �  r  �    N  h  n  b  5  �  �  �  �  (  r  �  F  	3  L  =  .    	  �  �  �  �  �  x  Y  <  2  (  )  ,  0  4  7    �  �  �  �  �  r  U  4    �  �  p  :    �  �  �  g  M  ,  +      �  �  �  �  Y    �  �  ;  �  �  -  x  �  �   �  �  �  �  �  �  �  �  �  �  p  c  S  B  -    �  �  j  �  j  y  d  O  :  -  !         �  �  �  �  �  �  �  �  �  f  K