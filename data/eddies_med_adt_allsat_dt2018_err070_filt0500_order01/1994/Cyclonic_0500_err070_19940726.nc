CDF       
      obs    P   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type         ����         	amplitude                   comment       �Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM���   
add_offset               min       ?h�t�j~�   max       ?��+J     @  �   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       M�pH   max       Pa�     @  �   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       ���   max       <��     @   ,   effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @?�Q�   max       @E�(�\     �  !l   effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ?���
=p    max       @v��Q�     �  -�   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @.         max       @P            �  :l   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @�e        max       @��          @  ;   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       ��+   max       <���     @  <L   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A��   max       B-h     @  =�   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A���   max       B,��     @  >�   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       >��W   max       C���     @  @   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       ?^�   max       C���     @  AL   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max          b     @  B�   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min             max          7     @  C�   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min             max          5     @  E   
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       M�pH   max       PM�     @  FL   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6��C-   
add_offset               min       ?�f�A��   max       ?��(��     @  G�   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       ���   max       <��     @  H�   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @?�Q�   max       @E�(�\     �  J   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ?���
=p    max       @v�fffff     �  V�   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @$         max       @P            �  c   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @�e        max       @�F`         @  c�   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         ?�   max         ?�     @  d�   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6��C-   
add_offset               min       ?�n��O�<   max       ?��t�k        f,               J                              
   
                              	         	            
         	   	   
                           	         :                      	      b                   %   .                     4   
      +            N��3M�pHO5g�N�x�PA�YNS�:N�4N�T�P��OIK�O��O)^�Nߊ�N��O	�(O7�`O�kHN�G�O��/N�0O®�O8_�O��hN�aONL�N�8TO��Ob�jO)9�N�5%O�N�&BN�1N���N&�(N�HN:��Nw�SO�F�N�#�O-\HO���O�gO`�N��+N(D�N�;"O~��N���Pa�N��gN�%�OC�O�ҸO\_�O���N��aN�GP��OT��O���N�n�OA[�O@vOߎ�O�XvN:VO�8*O,��OW��Nf��NK�O�RN1��OL45OsLO@��Nt��ND��N���<��<�h<o:�o%   �D�����
���
�ě���`B��`B��`B�o�t��#�
�49X�T���T���T���T���u��o��o��C����㼛�㼛�㼣�
��1��j��j���ͼ��ͼ�������������/��/��h�o�+�C��C��t��t��t��t��t���������w��w�''0 Ž0 Ž8Q�L�ͽP�`�P�`�Y��e`B�e`B�ixսixսm�h�m�h�q���q���q���u��t����
���T�� Ž�-��E����ͽ����������������������xz{�������zyxxxxxxxx��#),0)"������������������������'6B[ht����z\ROK6{{|���������}{{{{{{{#$/;<A<@<//#JU_abikiba_UNJJJJJJJD[t����������t[NBB<D��������������������HO[gt��������tg_[NFH#/<AHLI<:2/+#!)5865)''~�������������||~~~~<BGOSUTW[ahmh[OB=:9<��������������������/34:;@HQaeljYTH;,*(/�������������}�������������������������������"�����;HTamvwteaTHC<95003;��������������������}��������������{rsw}����������������������������������������ABNR[g�����tlg[NB>=A��������������������[gt��������tjg`[WTT[")5BNR[_b`^[NB@3-+)"|������������|||||||BBO[chotmha[[ODB==BB#)6BGJLB6)'"########��������������������qz�������zwoqqqqqqqq9BLOSSOB@99999999999�����������������������������������)67=:6)% #04DUbprmVI<630(& &)46;BLOSW[[OB76+)"&	#)0<CBA<0-'#
	�������("�������������������������������������������������������������������
����������������������������ainz�����������zna_a��������������������������������������������������������������������������������V\gqt���������th[YUV����������������������
#+/#
������)74752,)�����KO[`cckstttqh[TOLCGK��������������������#<M]adktyvmUH</##mqt{������������zpnm����������������������������������������
#&%#$#!
���(0<IOSTTROIF<10-*''(��������������������#0<IUflnlg_I<0(#������������������������
#/250#���������������������������������������~}|~������������������������������������������,/<Hao~trnmibUQH>/,,#)/5/)!��������������������[_bgjrt����������g[[�����������������������)20)CHU]afnnnhba^UOHFACCE�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E����������������������������������������������$�'�0�;�=�I�R�V�W�V�S�I�;�0�$�������!�-�7�:�F�K�F�@�:�-�!���������p�k�j�n�o�x���������	��!�	���������Ľý������������ĽнҽڽнĽĽĽĽĽĽ����t�s�o�s�t���������������������������������������������������������������������ݿɿϿڿ����(�5�G�F�1�6�5�'�%�������H�;�0�;�?�H�T�U�_�a�n�m�z�����z�m�a�T�H�	��������������������� ��	���� ��	�U�M�H�@�<�E�H�P�U�a�x�zÇÎÇ�~�z�n�a�U�/�(�&�'�.�/�;�<�F�H�T�\�Z�T�H�;�/�/�/�/���������������	��"�%�$�"� ��	����������s�f�\�f�s��������������������������������������������������������������������������������������	�"�;�H�M�J�;�/�������������������������������������������������������$�1�2�.�$����ƸƳƩƧƥƧƧƯƳ���������������������������(�5�Z�k�������s�Z�N�5�(���g�f�Z�Q�O�Z�g�s�������������������v�s�g�Ŀ������������Ŀѿݿ����"�,�0����ݿ����
�	��	����"�+�/�;�C�H�H�;�/�"�Ɓ��u�h�\�Y�\�d�h�uƁƃƈƂƁƁƁƁƁƁ�����������|�����������������������������M�B�A�8�A�M�R�f�s���������������s�f�M����������	��"�'�4�4�.�,�"���	�����������������������ʾ׾ݾ�پ׾ʾ������������������������������������������������H�G�<�B�H�L�T�a�m�y�z�z�z�u�m�l�a�T�H�H�m�h�b�g�m�y�����������y�m�m�m�m�m�m�m�m�����{�x�l�k�c�e�i�l�x�{�����������������ܻػӻѻܻ������������ܻܻܻܻܻܻܻ��U�I�U�W�a�n�q�o�n�a�U�U�U�U�U�U�U�U�U�U�ݽ۽ν̽нݽ���������"�������ݻ�ݻ�����	���������������ìççììù��������üùìììììììì������!�(�-�:�F�_�i�g�_�S�F�:�-�!��_�\�S�R�S�U�_�c�l�x�������������x�v�l�_�û����������ûлܻ���������ܻһлÿ`�W�G�<�@�G�T�V�_�k�m�y�����������y�m�`�3�'�����'�4�@�M�Y�f�p�r�i�f�Y�M�@�3���
���
���#�+�0�2�<�U�]�Y�W�I�<�0��������������������������������������������{�s�p�q�s����������������������������Ň�~ŇŐŔŠŧŭŹźŹŹųŭŠŔŇŇŇŇ�7�'�!���'�3�@�Y�c�e�g�g�g�d�d�g�\�L�7�C�;�9�6�5�6�C�O�\�h�i�h�e�\�X�O�C�C�C�C���ۺֺ˺ֺ���F�����ûǻǻ������_�F��~�t�w�~�������������ź��������������~�~ìèãåìù��������ÿùìììììììì�Ŀ������������������Ŀȿѿ׿���ݿѿ�ŵūũŭŹž��������������������������ŵ���������������&�)�6�7�+�+�)�'������нͽؽ������(�4�8�@�B�5������ݽйù����ùϹܹ�������� �������ܹϹ�ŔŊŌŔŠŭűŭũŠŔŔŔŔŔŔŔŔŔŔE�E�E�E�E�E�E�E�E�E�FFFF]FaFVFAF1FE�E�@�4�'�������'�4�@�Y�`�Z�^�Y�M�G�@�d�b�m����ʼ�����!�$����］����d�3�*�)�+�3�@�L�Y�`�e�r�r�r�l�e�Y�L�H�@�3�Ľ������������������Ľнݽ����߽нĽ���������(�4�A�J�M�T�M�L�4�(�����������������������
�#�E�S�Y�_�U�<�#��}�s�o�n�p�����������������������������}�t�n�l�g�b�g�t�v�t�t�t�t�t�t�t�t�B�5�)������������)�=�E�I�J�N�W�_�[�B��������������)�6�B�E�B�>�6�)������s�m�m�r�tĉčĒĚĦĵĿ����ĿĳĚčā�s�t�n�p�tāčĚĦĚėčā�t�t�t�t�t�t�t�t��	������߹����������������������������׾�����"����㾾������}�r�n�r��������������������������������������z�x�v�x�|���������������G�<�/�!����������
��#�/�<�H�K�M�I�J�G�ʼü��������������ʼμּ�����ݼּʽG�G�<�:�6�:�G�S�]�`�c�`�S�G�G�G�G�G�G�GÇÇ�z�u�s�zÇÎÒÓÔÓÇÇÇÇÇÇÇÇ�������������������Ŀѿҿѿ̿Ŀ��������� ? n U 2 ) 2 + . d = � ' 4 ) p M e 1 J = X / E N N � j " = : + 0 v : F p H Y N ? ; K c \ x < L @ - j z < H 5 Z : m L D 4 | P ' J R  ] n R I Q ` U [ L ; 4 . P D  �  ]  �  �  g  Z  �  �  �  �  �  v  �    �  �  /  �  '  &    �  S  �  �  �  �  �  �  �  $  �  V  �  D  2  T  �  4    |  Q  v  s    O  �  !  �     �  �  �  2    O  5  -  �  �  o     �  �  A  �  ]  �  �  �  �  �    H  �    �  |  j  �<�C�<�����o���
���P�t��D���#�
�t�����e`B��h�D���e`B���
��1�����
��㼼j��P��h�<j��󶼼j��`B���49X����h��P���C��o��P�\)�\)�t��D���]/�49X�]/�,1�@��'�w�8Q콁%�L�ͽ\�8Q�T���L�ͽq���}󶽗�P�T���@���+��O߽��罍O߽�{��������񪽃o��{��hs������%������m��Q���`�o��h�ě��������B�nB�DB��B,��B��B)V�B,�B�eB	�BFB	��B ;BO�B �]B�5BA��B �yBa�B9A�eB]�B*KB!X�B�+B	*�B ��B	�`BՂB�BB�=B!�B��B`?B!A�B��B*jB&��B�B%HjB�eB"��B�BT�B�>B�DB��B��B-�B-�B��B=�B��B�WBu4Bu"B��BF[B)w�B-hBʃB$Y�B&)�B<B&x�B�MB�BFB�BnB!>B��B.�B05B
:�B �B�Bd%BբB�ZB�GBG�B,��B�MB)?�B>�B�B	��B>�B	��B=�BE'B � B��B �A���B ��BAQB=XA���B<�B*�B!@�B��B	?�B �wB	�ZB?�B(�B5�B��B ��BàBEB!6�B��BF�B&�
B@B%L�B	lB#:�B � B�UB�B��B��B�;B=B?�B��BfB��B��B�\B>�Bz�B�_B)R�B,��B�KB$@�B&?�B<AB&I�BA�BD`B>�B8�BPB![�B��B?�B�B
>�B<�B��Bz�BàC�(A�	iB
n1@r�|A�foA&�A���A�tyA��A�`�A� !A�-wA���A�~:AGj�A��A�+A�
PB}�B�rA�4A��qA~�	A��MB�A�V2AB�iA[��AN҆A��;A�'�Al��@���@��fA�Q*A.�z@�DxA�f�@w]@�;V@�ѩAj��@�ٚA믩A��	A�`<A�q�?���B8u@���@ �A�BzAx�A���A�I8A2�>��WA�>�C���@��z@�y?��	A&j�A5�HA�l�A� �A���A��cAՒ�A�{�A� <?=��AR��@�^A���A��a@�ؔA��A�ioAv5�C� %A���B
@O@x9A��,A&e�A��0A�z�A��TA��!A�y�Aň�A��A���AI�/A��9A��A�D�B��B�eA���A��A}��A��9B7�A�|oAA"�A[&AP��A�[!A�+�Al@�V�@���A�{�A,�=@�W2A͂Z@v��@�>6@�!�Al��@�6A�A�dA��A�`�?��B@v@���@xA�n�Ax��A��A҆#A3��?^�A��C���@Χ�AZ?�wA%��A5!�A�k�A�f�A�| A�v$A�6vAޑ�A݌�?6
9AR�C@��A�A���@��A��A�^�At��               K                                                               
         
            
         
   	   
               	            
         :                      
      b      !            &   /                     4         +                           /            /                        %            #      '                                                                                 7                  !         )      3            %         #               %                                    %                                    %            #                                                                                       5                           '      3            !                                             N��3M�pHOdN�x�PVnNS�:NJ��N�T�O��ORO��N���Nߊ�N��N���O7�`O�kHN�G�OKF�N�0O®�O)WO���N�aONL�N�8TO	��O	O)9�NU�;O�N�&BN�1N���N&�(N�HN:��Nw�SO�F�N�yN��CO���O�gO`�N��+N(D�N�;"OeŸN���PM�NB�N�%�OC�O~�}O*dO�6N��aN�GP�YOT��O���N8��O6�O@vO��O���N:VOLi�OAUODJ|Nf��NK�O���N1��OL45OJ�`O4F�Nt��ND��N���  �  b  m  �    :  �  �  �  �  �  �    �    �  �  D  <  "  Z  D  D  �  �  >  �  �  c  �    Y  2  �  8  c  �  �  k  e  �  �      M  �  z  �  �  �  �  |  m  L  �  �  �     v  �  �  �  P  �  �  -    �  j    b  >  �  �  �  �  �  _    <��<�h;ě�:�o��C��D���ě����
�e`B�#�
��`B�D���o�t��49X�49X�T���T����C��T���u��C����ͼ�C����㼛�㼣�
��/��1�ě���j���ͼ��ͼ�������������/��/��h�+���C��C��t��t��t��t������,1�#�
��w��w�,1�49X�<j�0 Ž8Q�]/�P�`�P�`�q���ixսe`B��%��%�m�h��o�u�u�q���u���罣�
���T��Q콴9X��E����ͽ����������������������xz{�������zyxxxxxxxx� !(),)�������������������������'6BO[r|}zth[O6)"{{|���������}{{{{{{{#)/6<><9/#JU_abikiba_UNJJJJJJJJNR[t��������thNLHEJ��������������������HO[gt��������tg_[NFH#/9<EA<<///#)5865)''~�������������||~~~~>BEJORTSV[]`d[OB>;;>��������������������/34:;@HQaeljYTH;,*(/�������������}�������������������������������"�����;HTamvwteaTHC<95003;��������������������z���������������{zyz����������������������������������������ABNR[g�����tlg[NB>=A��������������������[[gt��������thg][Z[[")5BNR[_b`^[NB@3-+)"~����������~~~~~~~~~BBO[chotmha[[ODB==BB#)6BGJLB6)'"########��������������������qz�������zwoqqqqqqqq9BLOSSOB@99999999999�����������������������������������)67=:6)% #04DUbprmVI<630(& ')568BKORVYOB86,)#''#-08950,#�������("�������������������������������������������������������������������
����������������������������adjnz����������zna`a��������������������������������������������������������������������������������V\gqt���������th[YUV������������������������


������ )42430)��� KO[`cckstttqh[TOLCGK��������������������#8HUanvtlbUH</##mqt{������������zpnm����������������������������������������
#%$""
����(0<IOSTTROIF<10-*''(��������������������&0<IU`gihcZI<0#����������������������
#+/121,#
��������������������������������������}}����������������������������������������8<HVkmnkhfdaUHEA<438#)/5/)!��������������������egilpt����������tgde������������������������)20)CHU]afnnnhba^UOHFACCE�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E����������������������������������������������$�*�0�=�A�I�M�U�P�I�H�=�8�0�$�������!�-�7�:�F�K�F�@�:�-�!�����������{�u�v�t���������������������������Ľý������������ĽнҽڽнĽĽĽĽĽĽ����x�s�q�s�z�����������������������������������������������������������������������ݿ׿׿ݿ�������&�$���������H�>�C�H�M�T�a�m�p�z���z�m�a�T�H�H�H�H�	��������������������� ��	���� ��	�U�S�H�D�B�H�K�U�a�n�n�z�{�z�v�n�a�Z�U�U�/�(�&�'�.�/�;�<�F�H�T�\�Z�T�H�;�/�/�/�/���������������	��"�%�$�"� ��	����������v�s�f�_�f�s������������������������������������������������������������������������������������	�"�;�H�M�J�;�/������������������������������������������������������$�.�/�*�$���ƸƳƩƧƥƧƧƯƳ���������������������������(�5�Z�k�������s�Z�N�5�(���s�g�Z�S�P�Z�\�g�s�������������������s�s��ѿ����������Ŀѿݿ��� ������������
�	��	����"�+�/�;�C�H�H�;�/�"�Ɓ��u�h�\�Y�\�d�h�uƁƃƈƂƁƁƁƁƁƁ�����������|�����������������������������M�E�A�A�M�V�f�s����������������s�f�Z�M������������	���"�*�,�"�!���	�����������������������ʾ׾ݾ�پ׾ʾ������������������������������������������������H�G�<�B�H�L�T�a�m�y�z�z�z�u�m�l�a�T�H�H�m�h�b�g�m�y�����������y�m�m�m�m�m�m�m�m�����{�x�l�k�c�e�i�l�x�{�����������������ܻػӻѻܻ������������ܻܻܻܻܻܻܻ��U�I�U�W�a�n�q�o�n�a�U�U�U�U�U�U�U�U�U�U�ݽ۽ν̽нݽ���������"�������ݻ�ݻ�����	���������������ìççììù��������üùìììììììì������!�(�-�:�F�_�i�g�_�S�F�:�-�!��_�\�S�R�S�U�_�d�l�x�����������x�t�l�_�_�û����ûŻлܻ����ܻջлûûûûûÿ`�W�G�<�@�G�T�V�_�k�m�y�����������y�m�`�3�'�����'�4�@�M�Y�f�p�r�i�f�Y�M�@�3���
���
���#�+�0�2�<�U�]�Y�W�I�<�0��������������������������������������������{�s�p�q�s����������������������������Ň�~ŇŐŔŠŧŭŹźŹŹųŭŠŔŇŇŇŇ�;�3�'�#���(�3�@�L�Y�c�e�e�b�b�e�Z�L�;�C�;�9�6�5�6�C�O�\�h�i�h�e�\�X�O�C�C�C�C�F�����ݺ���F�_�������ûĻ����l�S�F�~�z�{�~���������������������~�~�~�~�~�~ìèãåìù��������ÿùìììììììì�Ŀ������������������Ŀȿѿ׿���ݿѿ�ŶŬŪŮŹ����������������������������Ŷ������������������������)�&�������ݽ׽ѽڽ������(�5�=�?�4�.������ݹù����ùϹܹ�������� �������ܹϹ�ŔŊŌŔŠŭűŭũŠŔŔŔŔŔŔŔŔŔŔE�E�E�E�E�E�E�E�FF$FCFZF_FTF?F1FFE�E�@�4�'�������'�4�@�Y�`�Z�^�Y�M�G�@�d�b�m����ʼ�����!�$����］����d�3�0�3�;�@�L�Y�^�Y�R�L�@�3�3�3�3�3�3�3�3�Ľ������������������Ľнݽ���߽ݽнĽ���������(�4�A�J�M�T�M�L�4�(�����#��
���������
��#�0�=�I�N�T�V�V�I�<�#�����x�t�r�v�����������������������������t�n�l�g�b�g�t�v�t�t�t�t�t�t�t�t�)����������������)�7�?�@�?�@�5�)�������������)�6�B�C�B�A�<�6�)���čąā�t�n�o�tČčĎĚĦĳĿ����ĿĳĚč�t�n�p�tāčĚĦĚėčā�t�t�t�t�t�t�t�t��	������߹��������������������������ʾ����	�������׾�������}�r�n�r��������������������������������������z�x�v�x�|���������������/�)�#��
�������
��#�/�<�H�I�K�H�A�<�/�ʼļ����������������ʼּ̼����ּܼʽG�G�<�:�6�:�G�S�]�`�c�`�S�G�G�G�G�G�G�GÇÇ�z�u�s�zÇÎÒÓÔÓÇÇÇÇÇÇÇÇ�������������������Ŀѿҿѿ̿Ŀ��������� ? n X 2  2 $ . ' . �  4 ) h M e 1 F = X / ? N N � h ( = 4 + 0 v : F p H Y N < P K c \ x < L ) - b T < H , < : m L < 4 | ] # J G  ] Q @ L Q ` R [ L : 8 . P D  �  ]  L  �  8  Z  V  �  :    �  �  �    2  �  /  �  �  &    o  k  �  �  �  ~  6  �  x  $  �  V  �  D  2  T  �  4    �  Q  v  s    O  �  �  �  +  t  �  �  �  f    5  -  �  �  o  ~  �  �    >  ]  �  q  �  �  �  D  H  �  �  �  |  j  �  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  �  �  �  �  r  S  4    �  �  �  �  X  /    �  �  u  @    b  `  ]  [  Y  W  T  S  Q  P  N  M  K  F  9  ,         �  ]  e  j  l  f  ]  S  I  ?  2  !    �  �  �  Z  '  �  �  �  �  �  �  �  �  �  �  �  �  �  �  ~  n  \  G  2       �   �  <  �  �          
  �  �  �  �  B  �  �  .  �  �  �  <  :  3  -  '      
  �  �  �  �  �  �  �  �  o  \  I  7  %  �  �  �  �  �  �  �  �  �  �  �  �  �  �  k  O  4     �   �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  Y  T  u  m  z  �  �  �  �  p  a  N  5    �  �  v  5    �  �  �  �  �  �  �  �  �  �  �  �  �  �  l  L    �  �  �  r  �  �  �  �  �  �  �  �  �  y  c  L  1    �  �  �  �  A   �  ^  t  �  �  �  �  �  y  d  H  "  �  �  �  s  u  �  i  H  #      �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  x  j  \  9     �   �        �  �  �  �  �  �  �  �  �  �  �  Z  .  �  �  i    �  �  �  �  �  �  �  �  w  f  S  ?  +    �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  z  c  I  '  �  �  a  D  8  -      �  �  �  �  �  �  j  P  6    �  �  �  �  w    *  7  <  1      �  �  �  �  �  `  8    �  S  �  `  �  "            �  �  �  �  �  �  �  �  �  _  6  
  �  �  Z  M  ?  .      �  �  �  �  s  R  4    �  �  �  3  �  ?  D  D  A  <  5  *      �  �  �  �  �  X  *  �  �  d     �  �    (  7  @  D  C  8  #  
  �  �  �  m  /  �  �    �    �  z  m  Y  9     
  �  �  �  �  �  b  E  -    �  �    ]  �  �  �  |  v  q  n  k  h  e  l  ~  �  �  �  �  �  �  �  �  >  .           �  �  �  �  �  u  k  X  4    �  |  4   �  �  �  �  �  �  �  o  Y  B  (    �  �  z  /  �  �  =  �  R  7  Y  q  �  �  �  �  �    p  _  J  %  �  �  b    �  W  �  c  _  Z  U  P  D  5      �  �  �  �  g  G  #   �   �   �   Z  �  �  �  �  �  �  �  �  �  {  t  l  d  \  R  I  =  (    �      �  �  �  �  �  �  �  v  X  5    �  �  �  Y  *  �  m  Y  S  M  G  C  ?  <  8  5  2  /  ,  )  #    
  �  �  �  �  2  ,  &        �  �  �  �  �  �  v  d  X  O  K  F  ?  7  �  �  {  n  _  P  A  0    
  �  �  �  �  i  <     �   �   s  8  F  S  `  g  i  e  ]  E  $  �  �  �  Q    �  �  ]    �  c  Y  O  A  4  #    �  �  �  �  �  |  d  V  F  /    �  i  �  �  �  �  �  �  �  q  Z  B  +    �  �  �  x  F    �  �  �  �  �  �  �  �  �  �  �  �  �  �  `    �  8  �  �  !  �  k  `  X  O  E  9  *      �  �  �  �  �  u  @    �  �  D  c  e  _  S  D  ,    �  �  �  \    �  �  T    �  a  �  �  {  ~  �  �  �  �  �  �  �  �    u  g  N  7  .  +  0  <  J  �  �  �  �  �  �  �  w  Z  5  	  �  �  r  9    �  �  �  �           �  �  �  �  �  ~  c  J  /    �  �  Q     �   �    �  �  �  �  �  �  �  �  �  �  �  t  m  c  Q  <  $  T  �  M  B  7  ,  !    	  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  y  p  d  W  I  <  .        �   �   �   �   �   z  z  p  f  ^  U  E  3    	  �  �  �  �  �  �  �  ~  m  �  �  �  �  �  �  V  "  �  �  a    �  �  @  �  �  B  �  H  �   �  �  �  �  �  �  �  �  �  �  �  �  �  w  g  K  '  �  �  m  �  �  �  �  �  Q  $  �  �  T  �  �  �  �  R    �  j  �  R  �  Y  W  V  \  t  �  �  u  `  >    �  �  �  y  K    �  �  s  |  z  x  u  o  c  Q  5    �  �  �  Y    �  �  ;  �  �  )  m  j  h  g  e  b  ]  X  O  E  8  (      �  �  �  �  {  T  I  L  J  B  <  7  3  +         �  �  �  �  ^  !  �  �  9  �  �  �  �  �  �  �  �  �  �  }  X  %  �  �  z  7  �  y  m  �  �  �  �  �  ~  h  Q  7  !    �  �  z  ?     �  n  $    �  �  �  x  e  O  :  "  	  �  �  �  �  �  �    .  8  �  �                   
        �  �  �  �  �  �  �  �  �  p  R  +    �  �  �  �  g    �    
n  	�  �  �  q  �  t  �  �  �  �  �  �  x  `  E  )  
  �  �  �  w  J    �  �  �  �  �  �  _  5  
  �  �  �  s  5  �  �  �  B  �  |  @  D  �  �  �  �  �  �  �  �  �  �  �  �  `  =    C  %  �  �  i  #  E  P  H  5      �  �  �  �  l  D    �  �  Y    �    H  �  �  �  �  �  �  k  M  .    �  �  �  ^    �  >  �  J   �  �  �  �  �  �  �  �  �  �  i  ?  
  �  g    �  _  �  W    �    +  ,  #      �  �  �    V  +  �  �  y    z  �  �    �  �  �  �  �  �  �  �  �  o  @    �  �  �  T      �   �  _  �  �  }  �  �  �  �  w  R  (  �  �    9  �  �    �   �  I  ^  b  O  5    �  �  �  �  �  {  d  P  @  !  �  �  �  �  �  	    �  �  �  �  �  �  r  V  6    �  �  y  >  �  �  ,  b  ]  W  R  M  I  I  I  I  I  A  0       �  �  �  �  �  y  >  &    �  �  �  �  �  �  �  }  p  a  O  =  +        �  �  �  �  �  �  �  �  �  �  c  +  �  �  `  �  �    J  �    �  �  j  Y  H  5    �  �  �  e  )  �  �  �  I    �  �  _  �  �  �  �  \  5    �  �  �  k  ?    �  �    �  H  �  d  Z  �  �  �  �  �  �  `  =    �  �  e  �  t  �      �    �  �  �  �  �  v  U  )  �  �  �  D  �  �    �    Y  �   �  _  N  =  ,    
  �  �  �  �  �  �  �  �  r  g  \  �  �  �    �  �  �  �  �  �  �  �  �  �  �  x  k  ]  J  5       �    �  �  �  ~  S  (  �  �  �  k  4  �  �  _    �  <  �  s