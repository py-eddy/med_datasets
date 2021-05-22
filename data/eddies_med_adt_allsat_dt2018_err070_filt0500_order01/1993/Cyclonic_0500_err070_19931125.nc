CDF       
      obs    J   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type         ����         	amplitude                   comment       �Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM���   
add_offset               min       ?h�t�j~�   max       ?��+I�     (  �   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       N7�   max       P�B�     (  �   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       ��E�   max       <�     (  �   effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>�(�\   max       @E�p��
>     �  !$   effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ��z�G�    max       @vY\(�     �  ,�   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @'         max       @R            �  8D   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @�        max       @��`         (  8�   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       ��   max       <D��     (  :    latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A��   max       B,='     (  ;(   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A��   max       B,@     (  <P   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       >�X   max       C��j     (  =x   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       >��   max       C��c     (  >�   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max          Y     (  ?�   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min             max          9     (  @�   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min             max          1     (  B   
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       N7�   max       Pb��     (  C@   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6��C-   
add_offset               min       ?��ߤ?�   max       ?Ӭq���     (  Dh   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       ��E�   max       <�9X     (  E�   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>�(�\   max       @E�p��
>     �  F�   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ���Q�     max       @vYG�z�     �  RH   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @'         max       @R            �  ]�   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @�        max       @�          (  ^l   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         >�   max         >�     (  _�   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6��C-   
add_offset               min       ?��,<���   max       ?Ӭq���        `�                  -   $                              
                     )      
                     	   X      	                                       	   .                     '         (                     $                     O�Z1N��+P��Ns	N�;�OՍOPd�NE�tPa�O��NP�NB��N7�N,��O�+�N���Nw�mOp��O��P	�WN��LO�}OȾ�PINƆ?Nۚ�O{VxN�v�Ox��O@N�r�O9�yN��P�B�O�U*N:��O cN��N���N��mN��Nk��O3�2N�zDO�OD�JO'�rPT�NO�O�N%��N�Y�N6וO�)`O��O(z�O�%}N&cO�)@O���O�ĢO��INb��OGl�O�4N�$2O��O�؀N�M(N"�rO1NY�YNR�N��<�<�9X<e`B<D��<49X;o��o�o�D���D�����
�ě��ě��49X�e`B�u�u��o���
��9X��j�ě��ě����ͼ��ͼ���������/��h��h��h�����o�+�+�C��\)��w��w��w�#�
�',1�,1�0 Ž0 Ž0 Ž8Q�<j�@��@��T���T���T���aG��q���q���q���u�}�}󶽁%��o��o�����7L��7L��hs��hs�����������E�����������������������������������_amv����������zmfab_����

��������� )5<?=5)"����#,*/A8(%
� ����������������������#%(-#)5B[g�������tgG;53&)
"/<HTXYUI</#	����������������������ptu{���������tpppppp��


�������������>BIOTSOB:8>>>>>>>>>>����	��������#)/<BC@</-# ^hltz������thd^^^^^^ )5BEHHFB5) �������
������az�����������zm_]`[a��������������������)5BINX[\[NB5)'#��������������������OVmz�����yrbTLKJLMO��������������������"(/;DAF;/" �����!)+)����������������������sy}������������tqlms�����%	������W[hotttsjh[[OPWWWWWW+0<HUV[YURRJH<7/+((+DHUadhnz}}zngaUNHBDD[_ft������������tZX[�������������������#%-+'$#�����������������#%/<GEE?</$#������������������������������������������������������������
#'-&#
hlrt���������ztjheeh|�������������||||||)@GOT[aih[OB6)��)5A>53)����HNNZ[gtw����tg[NBHHSn|�������������g[NSU[hkjmh[RPUUUUUUUUUU��
 #%#"
������[amnwz~zmja^[[[[[[[[���������������rz���������zrrrrrrrrpt����������������tp����������������������������������������#0Ubdb^OF:0#

.028<IIID<00.-....../9IUbnso_XWNI<0).,-/��#$#����������		�������$)5BIQ[__ZRNB5)//2<FHSHG?<52///////�������������������z����������������zz�����������}|ABFN[grt����tg[NGBAA��������������������z{~��������������{z��������}RUamnvzzzzuniaYUSOORyz{|������zyuuuwyyyy�����������������������������������������
����������������
��#�/�=�A�D�<�#��
¦²·¼¹²¦������������������*�C�Q�d�X�G�C�6�*����������������������¼�����������������ƎƌƎƐƐƚƧƳƹƸƳƧƚƖƎƎƎƎƎƎ����²¦¦²���������
��	���������˿ݿѿ��������������Ŀ���!�"� ��������(�'�(�1�5�A�N�R�N�M�A�5�(�(�(�(�(�(�(�(���|�{������������Ŀ����ѿĿ���������	���	����"�/�;�H�T�`�e�e�_�T�H�/��A�6�4�2�.�4�A�C�M�N�N�M�A�A�A�A�A�A�A�A����¿²¯²·¿�����������������������˼M�H�M�R�Y�f�f�k�f�Y�M�M�M�M�M�M�M�M�M�M�U�J�U�X�a�n�t�s�n�a�U�U�U�U�U�U�U�U�U�U�H�;�"����"�;�H�T�a�m�z�������z�m�T�H�)�'�&�)�/�0�6�B�O�W�R�O�D�B�6�2�)�)�)�)�<�2�/�-�/�4�<�H�U�U�U�P�K�H�<�<�<�<�<�<����������������������������������������àÛÐÇ�u�d�[�_�n�zÇÓàâçó÷òìà�~ČĚĦħĳ��������������������ĳĦč�~�f�Z�]�f�r�����������������������r�f�f�����������������������������������������������������������������������������(���
���(�A�Z�g�s�������������g�5�(�����'�(�5�6�A�N�X�U�N�F�A�5�)�(���/�#�"�!�"�%�)�/�;�H�L�L�I�H�=�;�/�/�/�/���� ����$�0�=�I�V�[�\�Y�T�<�0�$�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D��b�V�I�=�0�'�(�0�=�I�V�b�o�uǉǋǈ�{�o�b��x�r�r�r�u�}�����������������������������������'�-�.�'���������Z�M�C�;�?�A�M�Z�k�s�������������s�f�Zùðóôù��������������������������ùù���[�A�7�;�;�L¦¿�������
���������U�M�F�A�I�Q�n�{ŇŔŠūŨťŠŔ�{�n�b�U�H�E�H�U�a�n�zÁ�z�n�a�U�H�H�H�H�H�H�H�H�I�H�=�4�0�.�/�0�=�I�R�V�b�b�i�n�b�V�I�I�$�#������$�0�=�@�C�=�;�1�0�$�$�$�$�����������������������������������������ƾʾ׾�������׾ʾ���������������|�����������������������������ùôìàÖÛàìùüýþùùùùùùùù�������y�{�����������ĽνѽнνĽ�������ÓÒÇÆÁÇÓàììíìààÓÓÓÓÓÓ�ܹԹֹܹܹ׹ܹ߹��������������������ܿ�����������	���"�-�2�3�9�5�.�"���������������	������	��������ŠŌ�v�n�f�n�Šŭ��������������߻����������û˻˻û���������������������E*EE!E*E,E7ECELEPE[E\EaE_E\E[EPECE7E*E*��������������������������������������������������� �	������������������Ŀ������Ŀſѿ׿տѿοǿĿĿĿĿĿĿĿĺr�f�Y�@�4�5�@�C�L�Y�e�r�~���������~�w�r�������������ɺ�����������ֺ����l�j�l�s�x�}�����������ûŻ����������x�l�������������������Ľнݽ�����нĽ���������������(�4�(�'�����������s�g�c�d�k�s���������������������������������Ľǽڽ������)�'�#�����н�ùíâåìù������������ ��������ù�l�a�\�Z�d�n�zÇÓàìùüúôêãàÓ�lD�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D������(�4�<�A�B�A�;�4�+�(���� ������������������������������������������������	��������	������������čĈā�u�{ĄčďĦĳĿ����������ĿĳĦč�����������������������
��#�*�$�����ؼ'�"����������'�,�4�@�B�@�7�4�-�'�û��ûǻлܻ�޻ܻлûûûûûûûûûù����������ùϹܹ�����������ܹϹù��������#�'�3�@�G�A�@�3�'���������������ʼʼּ����ּʼǼ���������������(�(�,�(�!����� ������� V L 0 T J 0 < K E & e < $ 0 T n J = X F g _ J u i H o + T G ( ; t N G j S 9 ) " L A 3 3 � R @ a 5 8 h t � : W c 9 o U ; d 0 D  c @ 3 5 7 G > � c   |  �  �  �  �  �  �  p  �  �  �  m    :  �  �  �  �  T  �    v  �  N  	    h  �  *  ]  �  �    �  �  �  !  �  �  �  �  �  �  �  �  �    �  W  <  p  �  �  �  Y  �  �  o  E  �  d  S  h  �  t  �  )  z  �  @  >  �  �  �%   <D����o;�`B<t��0 Ž�P�t��o�+�t��t��#�
��1�o��9X�ě��\)��w�,1��P���#�
��+�o�t��,1�P�`�D���D���,1�P�`�����e`B�'<j�H�9�<j�0 Ž49X�D���m�h�aG��u�q���]/��+�Y���j�L�ͽT���q�����w��C����w�Ƨ�o���P���ͽ��
��E����-�\��-������Ƨ𽛥㽕����ͽ��9X��"�B?�B�B p�B#�qBѯB�UB�zB}�B�kB)<B�<B��B#��BhmBu�B�B(IB��BւB ��B @�BnsB�]A�jwBA��B�B�@B
��B��B�}B�B.#B
��Bj0B��B|AB4	Bq�B .TB!�BuB�]B
��B\ B�B	AaB
MHBsnB�TA�I�B �B�7B EB!��B.B%�5B&lB&�B�yB�BmRB�B��B��Bf�B�CBB)��B)]�B`IBo8B,='B��B?�B�B y�B$>CB BB3BK�BB�B�BB?�B�2B��B#�qBCQBB�B�zB@�BMyB0�B �B @B~�B��A��jB?�A��BI�B��B
�B�EB��B��B�.B{�BĜB��B�fBV�B{�B @uB!6kB��B��B8�B@�B60B	HB
��Bv>B��A�y�B ��B�XB =bB!<�B@�B%VmB&?B&�B��B+�BA�B�BE�BC9BAZB�B��B)��B)C�B?�BR�B,@B��A��A�^A���@��B{[A�\zA~_3A�bAuA�G-A:x	A�$@ڋ�Aơ�A�	�A��ZA��A���A�4jA���@��A�8�A�6A�ZA��'A�"PB
��C�&�B��@�ng@�@A@��A��PA�ϰA�[6A�mqB^9B	��A�Y;AR�0AI�TA�jbA"��Aʾ�?��A]��AY��A�(�@�.TC��jB�VBq�Ay�R?��@A?�@�=0A%<yA4�A�^OA.�2AЦA��C�͞A5��A�� A��A�(A��@�;�@�\@>�X?�RF@��8A��"A��A�ȒA�,@��eB��A���A�z�A��oAt��A��<A;-A��@�1�Aƃ�A��6A�x�AÒ|A�b�A�JTA��@���A�{�A��%A��A�Q�A��B
��C�%�B3�@��@�/A@ڥA�z:A��oA���Ań�BD�B
�A���AR6AH��ÀA!u�Aʎy?eA^��AYTHA���@���C��cB�
B��Ax��?�+L@5@@��RA#�*A6��A��A0��A���A��C���A6��A��A��MA�p�A怚@ɓ@��>��?�1I@�^A��o                  .   %                              
                     *                           	   Y      	                  	                     	   /                     '         )                      %                              )         %   )      '                  !               )         %   -                              9                                          -                           !                                                                                                !               )            #                              1                                          +                           !                                                   O8��N��+O�|%Ns	N�;�OиOE!>NE�tO��@OA�NP�NB��N7�N,��O�G�N���Nw�mO]��O��P	�WN��LO�}O�a�O��;N��Nۚ�O{VxN�v�Ox��N�>�NqK$O,��N��Pb��O�U*N:��N���N��N���N��mN��N��O	�N�zDNqmOD�JO'�rO��,NO�N�#,N%��N�Y�N6וO�)`O��O(z�O�%}N&cO�)@Owg�O�ĢO�U/Nb��OGl�O�4N�$2O��O��N�M(N"�rO�qNY�YNR�N��      �      �  V  �    �  J  �  e  �  �  L  _  �  �  �  N  �  ?  �  �  h  �  �    G  �  `  +  	e    b  m  �  )  �  �  [  �  �  �  ,  �  k  ^    0  �  �  B  �  �      S  "  ^  �  2  �  �  �  �      H  �  �  H  /<�9X<�9X;��
<D��<49X��t���t��o�T���T�����
�ě��ě��49X�u�u�u��C����
��9X��j�ě���/����������������/��h���������D���+�+�\)�\)��w��w��w�,1�0 Ž,1�H�9�0 Ž0 Ž49X�8Q�P�`�@��@��T���T���T���aG��q���q���q����C��}󶽃o��%��o��o�����7L��hs��hs��hs���㽙�����E�����������������������������������pwz������������zmkip����

��������� )5<?=5)"����
#$'##

�������������������������#%(-#55B[gr~}�~ytg[NB=<65#/<BHMOJH</#��������������������ptu{���������tpppppp��


�������������>BIOTSOB:8>>>>>>>>>>������
 �������#)/<BC@</-# ^hltz������thd^^^^^^)5BDGGEB;)�������
������az�����������zm_]`[a��������������������)5BINX[\[NB5)'#��������������������SWamz�����zumaTQPRRS��������������������"(/;DAF;/" �����!)+)����������������������sy}������������tqlms���!�������Y[hlsqh[RRYYYYYYYYYY,/1<HUUZXURQHH</,)(,DHUadhnz}}zngaUNHBDDkt������������tgccek�������������������#%-+'$#������������������#%/<GEE?</$#������������������������������������������������������������##*$#gimtt����������tohhg|�������������||||||)06ABDB6)��)5A>53)����HNNZ[gtw����tg[NBHHUow|������������g[PUU[hkjmh[RPUUUUUUUUUU��
"
��������[amnwz~zmja^[[[[[[[[���������������rz���������zrrrrrrrrpt����������������tp����������������������������������������#0Ubdb^OF:0#

.028<IIID<00.-....../9IUbnso_XWNI<0).,-/��� �����������		�������&)5BHOY^^YPNB5)//2<FHSHG?<52///////�������������������z����������������zz�����������}|ABFN[grt����tg[NGBAA��������������������z{~��������������{z��������}RUalnuyyyonmaZUSOORRyz{|������zyuuuwyyyy�����������������������������������������
�����������������
��#�(�/�3�7�0�#��
¦²·¼¹²¦������������������*�6�>�J�K�F�6�*������������������������¼�����������������ƎƌƎƐƐƚƧƳƹƸƳƧƚƖƎƎƎƎƎƎ¿¶²­°²³¼¿��������������������¿�ݿѿȿ����Ŀѿݿ����������������(�'�(�1�5�A�N�R�N�M�A�5�(�(�(�(�(�(�(�(�������������������Ŀѿ׿ֿϿɿ����������"�"����"�(�/�;�H�J�U�X�X�T�P�H�;�/�"�A�6�4�2�.�4�A�C�M�N�N�M�A�A�A�A�A�A�A�A����¿²¯²·¿�����������������������˼M�H�M�R�Y�f�f�k�f�Y�M�M�M�M�M�M�M�M�M�M�U�J�U�X�a�n�t�s�n�a�U�U�U�U�U�U�U�U�U�U�T�H�;�"����"�;�H�T�a�m�z�������z�m�T�)�'�&�)�/�0�6�B�O�W�R�O�D�B�6�2�)�)�)�)�<�2�/�-�/�4�<�H�U�U�U�P�K�H�<�<�<�<�<�<����������������������������������������àÛÐÇ�u�d�[�_�n�zÇÓàâçó÷òìà�~ČĚĦħĳ��������������������ĳĦč�~�f�Z�]�f�r�����������������������r�f�f�����������������������������������������������������������������������������5�(�����(�5�A�P�^�g���������s�Z�A�5����(�)�5�;�A�N�S�S�N�D�A�5�(�����/�#�"�!�"�%�)�/�;�H�L�L�I�H�=�;�/�/�/�/���� ����$�0�=�I�V�[�\�Y�T�<�0�$�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D��b�V�I�=�0�'�(�0�=�I�V�b�o�uǉǋǈ�{�o�b��z�s�w�~������������������������������������&�%�������������Z�Q�M�C�<�@�A�M�Z�h�s�����������s�f�Zùðóôù��������������������������ùù�g�Q�F�C�F�N�[�t¦²�����������¿�g�U�M�F�A�I�Q�n�{ŇŔŠūŨťŠŔ�{�n�b�U�H�E�H�U�a�n�zÁ�z�n�a�U�H�H�H�H�H�H�H�H�V�I�I�=�6�0�/�0�0�=�I�P�V�`�b�h�m�b�V�V�$�#������$�0�=�@�C�=�;�1�0�$�$�$�$�����������������������������������������ƾʾ׾�������׾ʾ���������������|�����������������������������àÙßàìùúúùìàààààààààà�������������������������ĽɽϽɽĽ�����ÓÒÇÆÁÇÓàììíìààÓÓÓÓÓÓ�ܹٹڹܹ����������������ܹܹܹܹܹܿ�����������	���"�-�2�3�9�5�.�"���������������	������	��������ŭŠŎ�z�nŃŠŭŹ���������������߻����������û˻˻û���������������������E*E!E#E*E/E7ECEPE\E^E]E\EWEPECE7E*E*E*E*��������������������������������������������������� �	������������������Ŀ������Ŀſѿ׿տѿοǿĿĿĿĿĿĿĿĺr�f�Y�@�4�5�@�C�L�Y�e�r�~���������~�w�r�������������ɺ�����������ֺ����l�j�l�s�x�}�����������ûŻ����������x�l�������������������Ľнݽ�����нĽ���������������(�4�(�'�����������s�g�c�d�k�s���������������������������нɽŽ˽ӽ������"�!��������ݽ�ùíâåìù������������ ��������ù�n�a�^�\�e�n�zÇÓàìùúùòèàÓÇ�nD�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D������(�4�<�A�B�A�;�4�+�(���� ������������������������������������������������	��������	������������čĈā�u�{ĄčďĦĳĿ����������ĿĳĦč�������������������������
��$��������'�"����������'�,�4�@�B�@�7�4�-�'�û��ûǻлܻ�޻ܻлûûûûûûûûûù����������ùϹܹ����������ܹϹù����������#�'�3�@�G�A�@�3�'���������������ʼʼּ����ּʼǼ���������������(�(�,�(�!����� ������� F L 7 T J 1 ? K : " e < $ 0 T n J : X F g _ > p c H o + T C   < t I G j O 9 ) " L / / 3 Y R @ ] 5 7 h t � : W c 9 o U * d 1 D  c @ 3 - 7 G < � c   �  �  �  �  �  Q  �  p  '  �  �  m    :  x  �  �  �  T  �    v  ,  �  �    h  �  *  5  o  }    �  �  �  
  �  �  �  �  .  D  �  �  �    ~  W    p  �  �  �  Y  �  �  o  E  �  d  1  h  �  t  �  )    �  @  0  �  �  �  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  �  �  �  �        �  �  �  �  Z  )  �  �    &  �  b  v    �  �  �  �  �  �  �  �  �  }  b  G  ,    �  �  �  �  �  u  �  �  �  �  �  �  �  �  �  �  �  �  f  ;    �  �  f  (      �  �  �  �  �  �  �  �  �  }  q  c  V  H  :  ,                                              �  �  �  �    >  X  �  �  �  �  {    �  	  �  �  �  �   �  �  >  �  �  �    :  L  T  V  V  R  ;    �  �  F  �  �    �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �        	          �  �  �  �  }  Z  (  �  �  N  E    E  y  �  �  �  �  �  �  �  �  ~  \  :    �  �  o  5  �  J  A  7  .  %        �  �  �  �  �  �  �  �  �  |  k  Z  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  e  ^  V  N  F  >  6  .  &             �  �  �  �  �  �  �  �                   �  �  �  �  �  �  v  \  @  $  �  �  �  �  �  �  w  `  G  ,    �  �  �  �  �  �  ]  �  X  L  C  9  .  !      �  �  �  �  �  `  <    �  �  J    �  _  [  W  O  F  <  1  !    �  �  �  �  �  |  _  9    �  �  w  �  ~  x  q  h  \  M  :  %    �  �  �  k  1  �  �  �    �  �  q  W  4  
  �  �  s  B    �  �  ]  �  w  ^  F  9     �  p  W  Y  `  O  A  G  V  L  ;    �  �  �  p  B    �  �  N  M  E  5    �  �  �  q  G  !    �  �  �  �  N    �  �  �  �  �  �  �  �  �  �  �  o  X  B  ,      �  �  �  �  �  -  /  1  9  ;  2  *  $      �  �  �  �    _  K  '  �  �  x  {  ~  �  �  �  �  �  �  P    �    �  .  j    d  �  �  �  �  �  �  �  �  �  �  y  d  N  9  &    �  �  �  �  .   �  h  `  W  L  A  7  -  $      �  �  �  �  �  r  k  h  w  �  �  �  �  �  |  f  K  +    �  �  �  z  Y  "  �  �  s  -  �  �  �  �  �  r  c  Q  :    �  �  �  y  G    �  �  D  �  �        �  �  �  �  �  �  `  3    �  �  �  ]  -  �  �   �  %  D  G  E  B  0    	  �  �  �  �  ^  ,  �  �  \    �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  r  :  �  �  -  �  n  P  _  P  >  *    �  �  �  �  \  (  �  �  M  �  �  =  �    +    �  �  �  �  �  �  �  j  d  x  �  n  U  <  !    �  �  �  	  	N  	c  	b  	Q  	,  �  �  �  G  �  �    d  �  '  t  �  �    �  �  �  �  �  �  f  U  G  0    �  �  u  (  �  �  `  ^  b  Z  R  B  .      �  �  �  �  �  |  W  2    �  �  �  p  `  h  i  _  L  4    �  �  �  �  `  <    �  �  G  �  v    �  �  �  �  �  �  �  r  ^  D  (  	  �  �  �  y  R    �  `  )      �  �  �  �  �  �  �  �  �  �  m  \  O  B    �  y  �  �  �  �  �  �  �  �  z  q  e  U  D  4  #     �   �   �   �  �    z  v  p  h  `  X  M  >  .         �   �   �   �   �   �  8  <  @  H  S  Z  V  S  F  7  (      �  �  �  �  �  �    �  �  �  �  �  �  �  �  �  t  V  -  �  �  v  (  �  �  J    �  �  w  a  I  0    �  �  �  �  `  3    �  �  _    �  �      �  �  �  �     b  �  |  f  J    �  �  �    Z  �   �  ,  '           �  �  �  b  +  �  �  m  +  �  �  C    �  �  �  �  �  v  j  \  M  8  !    �  �  �  �  \  )  �  }    e  h  X  E  1    �  �  �  J    �  �  �  }  �  m    �  .  ^  H  1    	  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �    �  �  �  �  d  2  
�  
�  
7  	�  	  M  s  �  �  �    0  '          �  �  �  �  �  �  g  M  3     �   �   �   �  �  �  �  �  �  �  �  v  f  R  ?  +      �  �  �  �  l  E  �  �  �  �  �  v  ]  F  .  !       �  h  4  !    �  �  �  B  )    �  �  �  y  I    �  �  l  ?  #      �  }  �   �  �  �  �  �  y  ^  ?  "    �  �  �  h  5  4  &    �  g  �  �  �  �  �  �  q  N  )  �  �  �  N    �  a  9  �  �  N  �    �  �  �  �  �  �  z  d  I  (    �  �  Y    �  V  �  ]    �  �  �  �  �  �  q  d  b  `  _  J  '    �  �  �  �  �  S  F  9  (    �  �  �  �  �  �  ~  l  W  ;    �  �  |  X        !  "  !  !      �  �  �  ]  '  �  �    r  z  i  ^  E  +    �      �  �  �  �  �  y  _  ?    �  �  x    �  �  �  �  �  �  �  `  1  �  �  �  M  �  �  $  �  �  E  �  2  �  �  �  w  X  5    �  �  x  H  &  �  �  8  �  |     �  �  �  h  D  #    �  �  �  k  :     �  R  �  _  �  \  �  �  �  �  �  �  �  �  c  H  /  ;  <  &  �  �  v  &  �    i  �  �  m  U  @  +    �  �  �  �  �  c  8    �  �  e  "  �  �  �  �  �  ]  '  �  �  �  P  "  �  �  �  S    �  K  �  g  5  �              �  �  �  �  �  �  O    �  �    �  �        �  �  �  �  �  �  �  �  �  q  _  L  9  "  	   �   �  H  D  @  <  8  4  0  ,  (  %                  �   �   �  �  �  �  �  �  �  q  U  2    �  �  �  O  �  �    w  �  "  �  �  �  �  �  �  �  �  u  d  R  @  +      �  �  �  �  �  H  ?  5  ,      �  �  �  �  �  �  i  K  %  �  �  �  P    /    �  �  �  p  @    �  �  j  1  �  �  �  E    �  ~  