CDF       
      obs    F   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type         ����         	amplitude                   comment       �Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM���   
add_offset               min       ?h�t�j~�   max       ?��$�/       �   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       M��[   max       P�
       �   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       �ȴ9   max       <u       �   effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>��
=p�   max       @FE�Q�     
�   �   effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ?�(�X    max       @v�fffff     
�  +�   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @,         max       @N�           �  6�   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @�        max       @�M            7`   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       ���   max       <t�       8x   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A�S�   max       B0�7       9�   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A�|_   max       B0�?       :�   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       >͖r   max       C��       ;�   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       >�A   max       C�Ǩ       <�   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max          T       =�   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min             max          =       ?   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min             max          =       @    
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       M��[   max       P_��       A8   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6��C-   
add_offset               min       ?��vȴ9X   max       ?�n��O�       BP   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       ���`   max       <e`B       Ch   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>��
=p�   max       @FE�Q�     
�  D�   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ?�(�X    max       @v�=p��
     
�  Op   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @'         max       @N�           �  Z`   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @�        max       @�           Z�   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         E)   max         E)       \   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6��C-   
add_offset               min       ?x*�0��   max       ?�����     �  ]         T                     '                           
   	      *      4      -                  @         
               "      '         G   1         (   .                     +         '      6      "                  N���O[P�
N��N�nNWCO�^O"ļO�j�P��N�@P�9N64�N�&�N=�IOa+�O.%NS�ZN4��N��`NE��O��N�V<P1A>N�{�PѸP��O��lO��O,|ONh�KP�$O1��P^Nb �NSc^N��N���N��O��O�ֹO{m5ND(�N8��PdccP�aP-&N��Ow�O�ON��O��rO�+�P7�O�+NG<�O���O��N�O-O��qNj�TP18N�Z�Ot�O?��N|�O
Y�M��[N�4]NÈw<u�o�D���ě���`B��`B�o�t��t��t��#�
�#�
�49X�49X�T���T���T���e`B�u��C���C����㼛�㼬1��1��9X��9X��j�ě��ě��ě��ě����ͼ�����/��/��`B��`B�������\)�\)��������w��w�#�
�''0 Ž<j�<j�D���H�9�P�`�T���ixսm�h�q���q���u��o��+��hs������Q�ȴ9�ȴ9V[hntv|tlhf[UPVVVVVVwz{���������������zw#0UblsuzvnU<+-,"(),5BEOZ[_Y[][QOB6)(xz�����zxwxxxxxxxxxx���


������������$/<Mai]a~�nH/#��������������������>BN[_p������tg[MC=<>������
#&"
�������#+//49<<<41/--)$#����������������xy~����������������������� 

�������W[gtvvtg[VWWWWWWWWWW������������������������������������������		�������������������������������).5@BDB95)
)44) )6B[gt����zthOB6& �������������������� )5NUjt���xg[TG) NO[hpttth[TOONNNNNN<Uanz�������znaUH<9<HU^bkt����������naUHz���������������{wzz�����

�������� #<HOKHC?</&#RUZanz}ztna]UORRRRRR���������������������������������#3<Ib{����{nbU<0$#����������������������������������������������������.6;COS\_f\XOIC6.*+..EHHTWabgknma[TQLJHDEktz������������zmhgk����������������������������

����������������������������������������������#/:@EUz���paUH@/##����&-,���������BNUtx�������tg[NB;<BU[]gtu������{tgg[QUU|����������������|z|VXWXamz�������ma[URV16BOOPSRUONBB7621111 $$05<IbggdcYQI<0#  5;HTagmopokaTH;32/05
#0<IPZ^`^USN>0#
mnov{�����������{rnm�������������������������������������������������������z{}|�GOP[hqohhhd[VODFGGGG��BL[b[UJ>85)��ggnrtz��������tggggg����!.7:8)�������������������������������������������+5JN[gkmnqg[B5/)��������������������/6=BO[hjmihe[OOB6.//����������������	
!###
		 #',/7<BHJKHD<4/#  �����������Ŀѿڿݿ޿ݿؿѿĿ��������������������������������������������
����q�m�t�{�������A�M�U�s�f�M�(��ݽ����
���������������������
���� ���
ùòôù������������ùùùùùùùùùù��ھܾ����������������������<�/��
���
��#�<�A�H�U�^�l�~Å�n�U�<�z�w�n�m�k�n�s�zÇÐÓÙàãàâàÓÇ�z�ʾþ��ƾʾ׾����	������	�����ʾ���㾾���������žʾ׾��
��#�*�(�	���5�)�)�(�)�5�B�R�[�\�g�t�u�t�g�[�N�L�B�5�����������������׾��������	��׾��h�a�a�h�tāĊā�v�t�h�h�h�h�h�h�h�h�h�hìáàÓÏÎÓàìõùúùõìììììì�f�]�^�d�f�s�t�{�{�s�f�f�f�f�f�f�f�f�f�f�4�(�������4�A�L�M�Z�[�\�Z�W�M�A�4�������������Ľнݽ���ݽԽнĽ��������	��������	���"�+�"��	�	�	�	�	�	�	�	ǔǓǈ�}�{�s�{ǈǍǔǡǢǢǡǔǔǔǔǔǔ�m�l�f�k�m�u�y�����������������������y�m������������������������������������������������x�s�i�d�s���������������������������������ºȺɺֺݺ���ںֺҺɺ�����������������)�0�A�O�\Ɔƀ�u�h�U�G�6����~�y�{����������������������������`�G�8�3�:�A�Z�g�����������������������:�1�������������������"�/�H�R�L�E�D�:�����i�^�X�`�m�������ĿԿܿڿʿĿ����������������������*�9�E�Q�M�C�6�*���6�0�)�(�+�.�6�B�F�O�[�b�b�[�Y�[�b�[�O�6�����
�����$�(�)�-�)����������� ���'�@�Y�z�����������~�r�3����� �����	���"�.�9�;�B�G�=�;�.�"������|�q�f�^�c�s������������������������ùõìäãìù��������úùùùùùùùù���������������� �������������������������������������������������׾ξʾǾȾʾվ׾��������������׾������������������	��� ���	����������ƚƁ�v�h�`�b�h�}ƎƧƳ��������������Ƴƚ���������������$�0�3�9�9�7�3�0�$��������ݿѿĿ��������Ŀѿݿ������ ������������(�5�:�8�5�(�������������������������������������������������E�E�E�E�E�E�FF F$F=F|F�F�F�FoFVF5E�E�E���������ۼ���!�6�;�9�.�����ʼ�����������������������������"�#��������������������������������������������������6�)����������)�6�B�J�O�]�^�V�O�B�6ĳĦčā�tĂĔĦĿ�����������������ĳ�S�I�G�S�S�_�l�x�������y�x�l�j�_�S�S�S�S�_�S�:�'�!���'�-�:�F�S�l�x�������r�l�_�A�7�.�*�%�(�/�5�A�N�Z�t�������s�g�Z�N�A�λ������������ûл������'�.�,���μ@�8�4�'�"���'�4�@�M�W�f�k�f�Y�S�M�D�@��������������������������������快�����������Ŀݿ�������������ݿĿ������������������$�,�8�>�;�;�5�0�$����ϹĹù����ùϹܹ����������ܹϹϹϹ�Ç�z�n�[�S�H�B�H�S�a�nÓàõüûþûàÇ�<�<�/�#��#�(�/�7�<�H�J�H�F�@�<�<�<�<�<�l�`�J�8�/�C�S�`�l�������Ľٽ۽ѽĽ����l�g�d�Z�N�Z�f�g�s���������|�s�g�g�g�g�g�g�������_�S�_�l�x�������ûлڻݻһû�����ĦĤĦīĦĦĩĲĿ��������������ĿĺĳĦ��������������!�"�'�!�������ìçàÝÕÔÛàìù��������������ùìì���������������������������������������EEEEEEE*E7EAECEEEOECEAE7E*EEEED�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D� 5 S U C W F > 4 S K o 3 F 7 6 ) X i b s N S u F E 8 K F , ? e < < Q 8 H . 9 { / E . N U g o d z < Y ^ 0 =  * \ , . / b ^ M 6 r { 2 G z s T  �  {  �    4  �  S  g  �  �    g  \  �  [  �  r  q  h  �  {  2  �  0  �    �  �  G  v  �  �  �  l  w  t  �    !    T    x  Y  Y  �  �    �  �  �  ]  Z  1  .  �      �  y  �  �  �  \  A  �  H  W    �<t���t��� ż#�
�t��t���㼼j���@��e`B��㼋C��u��o��h��/��t����ͼ�����j�u���ͽ�t��\)��7L�Y��49X�aG��0 ż�h��9X�,1�,1�t����0 Ž#�
��㽃o�ixս��P�'#�
��/�� Ž���8Q콡����-�y�#��7L��\)����]/������vɽ�O߽���ě��}��l���7L�ȴ9��1���㽾vɽ�vɽ���hB�)B�vB&�oBX	B B#��B�&B�(B	E�B}B!�B C�B��B��B	0[B�ZB")A�S�B6�Bz�B�BijB"I�B�GB��B��BB*�B�B"NBo�B`�Bm\B'B�=B*>�B��B0�7A���B �B�pB�_B�B�xB^_B-�9B	��B	�pB2�A�G"BAEB&��A�N�B%�oB)?,B1B��B`nB>�BTBB

�BKBU'B�uB�/B>�B��B�DB�Bo5B\�BA�B&�jB@�A��PB#��B�6B�XB	q^B�;B<�B��B�FB�FB�B3`B"?�A�|_B?�B��B�BE�B"M$BI	B�yB��Bc�B)�2BB>�BA�B>�B?�B&zmB�TB*WWB�HB0�?A��B n�BQ�BʩB�VB�:BH�B.8^B	��B	˲B3XA�~B?WB&@�A��KB%��B)>�B�ZB�|B�gB?LB:uB
b�B�
BHSB�nB�BJlBCB�B?9B�|Ayg>A��A*��A��HA�~�AV�yA�:A�_hAV��AU�A�c�AS)pA�OA��GAA�NA8�4A&��A� IB�Ao�SA�A8A���@4�A��@�lqA��A�bWAr�qA��Aؕ�A�R�?��A^�sA��bA�SQA��AI��AU*sA�6CBe6B	%�A|uA�	AA���C��A9]A�VUA���A�6~A��@�D.@�g�A�Q@��>@��	B;Az�nB	7M>͖rA�u6A�"�AS/A�N�@���A�A@`�A̲�A�GC���C��Ay
�A��dA&g~A��BAέNAU��AďhAɁBAY�AT�A�,AT>�A܉�A̕�AB��A8F�A':A�kqB<�An�A�}fA�{`@3��A��@� A���A��tAs�A�K�A؀A�`�?��A_�~A�		A�c(A�}LAH��AT��A�x�B?�B	?�A|�WA�Z<A��rC�ǨA�A��kA���A�a�A�i@��@|$-A��@�	@�(XB?�A{B	�>>�AA�+AêJA��A�RN@��1A��@^A��A��C��C��         T                     '                              
      *      4      .                    A         
               "      (         G   1         (   /                     +         '      7      #                           ;            +         )      %                              )      1      '   +   #            )      '                  !               =   5   )         '            %                  '      +                                 +            )         !      #                                             !               %      '                                 =   /   #         '            %                        !                        NygTN�`�P�
N��N�nNWCO���N��Oj�_O�p_N�@O�E�N64�N�&�N=�IO�`O.%NS�ZN�;N��`NE��O�QN�V<O��)N�{�O���Oƭ�O�P�O<�GO�TNh�KO��O�P^Nb �NSc^N��HN���N��O�$�O�O6duND(�N8��P_��PJ�O�p�N��OP��O�ON�~O�51O�+�O�9MO�+N1|TO1��O��N�O-Ou�8Nj�TO��9No;@Ot�O?��N|�O
Y�M��[Niw�NÈw  ,  �    R  �  6  �  �  �  �    N  �  �  �  q  �  �  �  )  U  T  �  I  [  �     x  F  \    �  �  u  �  �    &  �  �    �    D  �  p  $  �  �  /  a  �  �  �  �    �  �    �  �  ,  �  Y    �  *  ^  �  �<e`B���
�\)�ě���`B��`B�49X�49X�T���u�#�
�D���49X�49X�T����C��T���e`B��o��C���C���P�����㼬1����h���ͽC����ͼě���`B��`B������/��/����`B���+�o�,1�\)����w��w�,1��w�49X�',1�49X�<j�H�9�D���L�ͽy�#�T���ixս�O߽q������y�#��o��+��hs������Q���`�ȴ9X[hlttxth[WRXXXXXXXX����������������}��#0<U]cghg_WIB=:/%#(),5BEOZ[_Y[][QOB6)(xz�����zxwxxxxxxxxxx���


������������ '&/<HUae[a{|H/#��������������������JN[gt������tg[SJDBCJ�����

���������#+//49<<<41/--)$#����������������{{����������������������� 

�������W[gtvvtg[VWWWWWWWWWW������������������������������������������		�������������������������������).5@BDB95)
)44)MO[hptvyxvtph[POGBCM��������������������")5BNT[_gf[NB5)#NO[hpttth[TOONNNNNNEIPUanz������znaUKEEuw�����������znhdfuz||���������������}z������

�������!#<HMJC?</'#RUZanz}ztna]UORRRRRR�����������������������������������#3<Ib{����{nbU<0$#����������������������������������������������������.6;COS\_f\XOIC6.*+..EHHTWabgknma[TQLJHDEluz������������zmjil������������������������

�������������������������������������������������#/9@DUz���oaUH@/##�����&-+��������BN[gt��������g[NF??BU[]gtu������{tgg[QUU}����������������~|}VXWXamz�������ma[URV26BNOPSROCB862222222"%%$07<IbfgdbUOI<0!"5;HTagmopokaTH;32/05
#0IVZZVPNH<0#
mnov{�����������{rnm�������������������������������������������������������z{}|�GOP[hqohhhd[VODFGGGG 	)5:A?;5)��� ggnrtz��������tggggg���)//)$�����������������������������������������������+5JN[gkmnqg[B5/)��������������������/6=BO[hjmihe[OOB6.//����������������
!
 #',/7<BHJKHD<4/#  ���������ĿĿѿտۿֿѿĿ�������������������������������������������������꽍�������������ݾ�(�4�;�4�(���н������
���������������������
���� ���
ùòôù������������ùùùùùùùùùù��ھܾ����������������������<�/��
������#�<�C�H�U�i�{�}�n�U�<�z�y�o�n�r�z�ÇÊÓ×àáàÝÖÓÇ�z�z�ʾȾоؾݾ������	�����	�����׾ʾ��׾��������ʾ׾���	������	�����5�)�)�(�)�5�B�R�[�\�g�t�u�t�g�[�N�L�B�5�������������������׾������	��׾��h�a�a�h�tāĊā�v�t�h�h�h�h�h�h�h�h�h�hìáàÓÏÎÓàìõùúùõìììììì�f�]�^�d�f�s�t�{�{�s�f�f�f�f�f�f�f�f�f�f�(�!��������(�4�:�A�M�R�W�M�A�4�(�������������Ľнݽ���ݽԽнĽ��������	��������	���"�+�"��	�	�	�	�	�	�	�	ǡǕǔǈ��{�z�{ǈǋǔǡǡǡǡǡǡǡǡǡ�m�l�f�k�m�u�y�����������������������y�m���������������������������������������������������}���������������������������������������ºȺɺֺݺ���ںֺҺɺ������*����������#�*�C�O�Z�]�X�R�I�C�6�*��~�y�{������������������������������s�g�O�@�<�A�N�g���������������������������������������	��"�;�B�B�8�/�"�	�𿫿����y�l�a�m�y���������Ŀ̿ҿӿѿƿ���������������*�1�6�?�D�F�C�6�*��6�1�)�)�,�0�6�B�O�[�a�a�[�Y�[�`�[�O�B�6�����
�����$�(�)�-�)��������������'�3�@�Y�u�������~�Y�@�3�����	������	����"�.�;�?�B�;�8�.�"������|�q�f�^�c�s������������������������ùõìäãìù��������úùùùùùùùù���������������� �������������������������������������������������׾ξʾǾȾʾվ׾��������������׾������������������	��� ���	����������ƚƁ�x�j�b�e�hƁƎƧ����������������Ƴƚ�� �������������$�0�2�9�8�6�3�0�$���Ŀ��������Ŀѿݿ�������������ݿѿĿ�������(�5�:�8�5�(�������������������������������������������������E�E�E�E�E�E�FF!F$F=F|F�F�F�FoFUF4E�E�E������������ܼ���!�4�:�8�.�����ʼ�������������������������������������������������������������������������������6�)��������)�6�B�D�O�Z�[�R�O�B�6ĳĦčā�tĂĔĦĿ�����������������ĳ�S�J�G�S�V�_�l�x���x�x�l�i�_�S�S�S�S�S�S�_�S�F�:�(�!���(�-�:�F�S�q�x�������l�_�A�7�.�*�%�(�/�5�A�N�Z�t�������s�g�Z�N�A�ܻû��������ûлܻ�����%�)�(�����ܼ@�8�4�'�"���'�4�@�M�W�f�k�f�Y�S�M�D�@�����������������������������������Ŀ��������������Ŀѿݿ�������ݿѿ˿����������������$�,�8�>�;�;�5�0�$����ϹĹù����ùϹܹ����������ܹϹϹϹ��z�t�n�i�a�_�`�a�n�zÇÔäëðëàÓÇ�z�<�<�/�#��#�(�/�7�<�H�J�H�F�@�<�<�<�<�<�`�Y�K�I�`���������Ľǽ½����������y�l�`�s�h�g�Z�Y�Z�g�h�s���������z�s�s�s�s�s�s�������_�S�_�l�x�������ûлڻݻһû�����ĦĤĦīĦĦĩĲĿ��������������ĿĺĳĦ��������������!�"�'�!�������ìçàÝÕÔÛàìù��������������ùìì���������������������������������������EEEEE*E7ECELECE?E7E*EEEEEEEED�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D� 6 l b C W F E 1 R H o 0 F 7 6 0 X i h s N 4 u 7 E ? C 4 ! : e 1 * Q 8 H  9 { + F  N U h n c z 2 Y R 2 =   * S + . / 6 ^ F 3 r { 2 G z ] T  �  !  �    4  �    &        7  \  �  [  I  r  q  u  �  {  U  �  ^  �  �  �  "  �  f  �  D  +  l  w  t  �    !  �  ,  x  x  Y  M  �  ?    �  �  �  ?  Z  �  .  S  q    �  �  �  �  ~  \  A  �  H  W  �  �  E)  E)  E)  E)  E)  E)  E)  E)  E)  E)  E)  E)  E)  E)  E)  E)  E)  E)  E)  E)  E)  E)  E)  E)  E)  E)  E)  E)  E)  E)  E)  E)  E)  E)  E)  E)  E)  E)  E)  E)  E)  E)  E)  E)  E)  E)  E)  E)  E)  E)  E)  E)  E)  E)  E)  E)  E)  E)  E)  E)  E)  E)  E)  E)  E)  E)  E)  E)  E)  E)  *  +  ,  ,  ,  +  *  (  %  #        �        �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �    B  �  }     ~   �  �  �  8  �  �  �        
  �  �  �  W  �  �    K  �   �  R  K  D  >  7  0  &      
     �   �   �   �   �   �   �   �   �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  6  7  8  9  :  <  =  >  ?  @  ?  ;  8  4  0  -  )  %  "    �  �  �  �  �  �  I  	  �  �  �  �    9  I  7    �  �  �  �  �  �  �  �  �  �  �  �  v  ]  @     �  �  �  Y    �    �  �  �  �  �  �  �  �  �  �  �  �  �  h  I    �  Y  �  '  �  �  �  �  �  �  �  �  y  @    �  s  4  +  �  m  �  �        �  �  �  �  �  �  �  �  �  �  �  t  e  P  7       �  C  L  M  K  G  I  L  M  K  E  9  (    �  �  �  \    �  �  �  �  �    x  r  k  c  Y  P  :    �  �  �  �  �  p  U  :  �  �  v  l  b  X  M  B  7  ,      �  �  �  �  �  �  `  ?  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  ?  Q  ^  h  o  p  m  f  [  K  6    �  �  �  �  P  ,  	  �  �  �  �  �  �  �  �  �  �  �  �  i  G    �  �  X   �   �   ;  �  �  �  �  �  �  z  s  l  e  ^  V  N  F  ?  (    �  �  �  �  �  �  �  �  �  �  s  G    �  �  �  Q    �  �  f  )  �  )  !          �  �  �  �  �  r  b  O  7    �  �  q    U  Q  N  K  F  A  ;  5  /  )  !        �  �  �  �  �  �  �  �  �  �        -  M  M  B  /    �  �  H  �       �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  v  h  V  D  2  �  d  �  �    :  H  F  ;  *    �  �  �  N  �  e  �  /  Z  [  K  :  *      �  �  �  �  �  ^  0  �  �  z  7  �  u  �    y  �  �  �  �  �  �  �  �  �  �  n  L  &  �  s  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  s  O  (  �  �  4  �  [  a  v  m  d  _  X  O  B  /    �  �  �  r  <  �  �  @   �      $  2  =  D  F  C  5    �  �  v  *  �  T  �  �    �  X  \  [  T  K  >  .    �  �  �  R    �  �  >  �  Y  �  �      �  �  �  �  �  �  �  �  �  �  ~  s  h  ]  :    �  �  �  �  �  �  �  �  �  �  Y    �  �  \  :    �  �  =  G  .  �  �  �  �  �  �  �  |  l  S  4    �  �  �  Y  (  �  �  o  u  \  A  '    �  �  �  �  �  �  �  �  �  �  �  g  >     �  �  �  �  �  �  r  V  ;      �  �  �  �  h  =  �  �  [    �  �  �  �  �  �  �  �  �  �  r  c  S  C  3  !     �   �   �    
        �  �  �  �  �  }  b  E  (    �  �     �  �  &  %  #      
  �  �  �  �  �  �  y  ^  A    �  �  �  4  �  �  �  �  �  �  �  �  �  �  �  o  ^  M  ,  �  �  x  &   �  �  �  �  �  �  �  �  �  �  �  �  |  ]  2  �  �  l    �    	  	  �  �  �  �  �  ^  0  �  �  �  C  �  �  f    �  ^  �  ^  �  �  �  �  �  �  �  �  �  R    �  d  �  �  
  �    �            �  �  �  �  �  �  �  �  q  S  5    �  �  ]  D  B  ?  <  :  7  4  2  /  ,  !    �  �  �  �  �  �  �  r  �  �  K  
�  
�  
(  	�  	|  	  �  ?  �  �  6  �  L  �    B  +  Y  f  O  8  %    �  �  �  �  z  N  )  �  �  J  �  C  |  :      $  #      �  �  �  �  �  �  �  ~  K    �  N  �  �  �  �  �  �  �  �  �  �  �  �  e  C  "        '  B  \  w  �  �  �  �  �  �  �  �  c  4    �  �  =  �  �    �  �  �  /         �  �  �  h  '  �  �  j    �  @  �  H  �    2  W  `  G  0       �  �  �  l  :  �  �  O  �  B  �  �    I  �  �  �  �  �  ~  j  Q  1  	  �  �  S    �  j  2    5  d  �  �  �  �  �  r  O  *     �  �  d  "  �  �  $  �  7  �  G  �  �  �  �  �  �  �  w  k  f  _  S  A  #  �  �    �  A   �  �  �  �  �  �  �  �  �  �  v  _  G  0              �   �  �    f  �  0  �  	)  	�  	�  
  
&  
@  
Y  
r  
�  
�  
�  
�  
�  
�  �    @  i    ~  l  N  -  	  �  �  ~  1  �  V  �  �  %  z  �  �  �  �  �  x  V  .    �  �  f  2  �  �  '  �  �  �  e           �    2  N  `  `  M  )  �  �  z  6  �  �  ^    ^  �  �  �  �  �  �  �  �  �  o  8  �  �  =  �  D  �    D  �  �  �    x  q  j  `  U  J  ?  4  )      �  �  �  �  �  �    "  &  )  ,  (    �  �  �  L  �  �  A  �  n  �  �  B    �  �  �  �  �  �  �  y  n  c  W  E  4  %      �  �  �  Y  ?    #  @        ,  5    �  �  [    �  O  �  9  �    �  �  �  {  H    �  �  �  �  {  F    �  ]  �  �     S  �  u  g  Y  K  ;  +      �  �  �  �  �  �  �  t  Q  /    *    �  �  �  �  |  i  J  !  �  �  q  (  �  �  E  �    �  ^  Y  S  M  H  B  =  <  >  @  B  E  G  J  O  T  Y  _  d  i  �  �  �  �  �  �  �  {  J    �  �  �  ?  �  B  �  5  �     �  �  �  �  c  8    �  �  �  W    �  \    �  d    �  �