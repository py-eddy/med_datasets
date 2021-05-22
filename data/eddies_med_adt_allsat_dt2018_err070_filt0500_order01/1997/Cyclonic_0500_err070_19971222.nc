CDF       
      obs    E   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type         ����         	amplitude                   comment       �Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM���   
add_offset               min       ?`bM���   max       ?���vȴ:       �   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       M�Q   max       P�[�       �   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       ��`B   max       =m�h       �   effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @?���
=q   max       @Fq��R     
�   �   effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ?У�
=p    max       @vz�G�{     
�  +�   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @/         max       @P�           �  6x   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @�        max       @�:�           7   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       �\)   max       =P�`       8   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       B �   max       B2��       9,   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A���   max       B2��       :@   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       ?�[   max       C���       ;T   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       ?1zI   max       C��       <h   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max          i       =|   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min             max          A       >�   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min             max          =       ?�   
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       M�Q   max       P��o       @�   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6��C-   
add_offset               min       ?��s�PH   max       ?��.H�       A�   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       ��`B   max       =m�h       B�   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @?���
=q   max       @Fq��R     
�  C�   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ?У�
=p    max       @vzz�G�     
�  N�   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @%         max       @O@           �  Y�   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @�        max       @��@           Z   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         Dr   max         Dr       [$   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6��C-   
add_offset               min       ?��$�/   max       ?��.H�     �  \8            N   h               3                  &                        ?               5         (            ?                         9         %                     	   	      5                           -   
      NU��N��N ��P�[�P�d�No�P��N�ZXN0�APy1N=vO�JN]Z�O��N�?DO� 4N"\�N#WN?gNC��N�pNN|�OO���P{ȁNa�O�N�7�N2O��N��N&~�P9!�NMH[N��.N��PST�N��{O;�N��mO��O�N�t�Nы-O�]~N�GWM�QO$�*O3Z�N��Of�P��N��Oo�N�^	N�fONcGO�G�N��%M�:�N�7�N#�CNd��N��nO`�O"�Omf	NH^N�N�i=m�h<�9X;��
:�o%   ��o�D����`B�49X�D���T����o���㼣�
��1��1��j�ě��ě����ͼ��ͼ������o�C��\)�\)�\)�t���P��P����w��w�#�
�#�
�8Q�D���H�9�H�9�L�ͽL�ͽT���aG��q���q���u�u�y�#��%��%��o��hs��hs���P���������������-���-����罬1��{�ě��ě��ě���/��`B#/<?=<0/(#jnvz�����������znkjj���������������������#<k�����\L<0������<UbjptynUI<0�����
#&#
�������0B[gt����t]NB5)�))-*)
��������������������������������������������068CGOXONC6-00000000ehity�����������xthe��������������������
#/3<HFA</#
MO[gd`_b[TOLJIMMMMMMmt���������������}sm��������������������pt������wvtpppppppp`hu����uha``````````�����������������������������������������������������������������������������BNg����������gNB=:;B()05>B95)'#$((((((((uz���������������wsuDHJU]abnu|znma[UMHDD_anz���znha________#/<H`hljwyvaUH</##��������������EO[hihc[OIEEEEEEEEEEaez��������������mba����������������������������������������������������1N[t�������g[NKB5$1�����

������������������������������ot{����������wtpjloo
)O[hlgYWSOB:0

R[ahntv����th[QOQSRR������������������������������������������#)"#''"
���������������������������������������������"#)/:<?HJMLJG</(#"!"��������������������rt~�����������ttrrrr������
!��������#0CO^`UR<0#��������������������������

��������qz�������������|zoqqmnuz���������zxrnnmmagnqnaUUTUaaaaaaaaaa������������������������� �������������������������������������������������������������������dgst������|tgeddddddIN[gt|�����tg_[NCII�
 ##� �����)5BKNXNBA52)&����������������������������������������stt~������������{tss��������������������ŭũŦŧŭŹŻ��������Źŭŭŭŭŭŭŭŭ�������������������������������������������������������������������������������������Q�B�5�*�Y���������������	����������y�l�l�o�v��������9�D�E�:����н��������������������������������������������N�5�$�����(�5�<�H�g�������������g�N�ɺȺɺɺɺֺֺ��������ֺʺɺɺɺ���������������������������������������꿆�m�G�4�1�;�T�y���Ŀѿ�������ѿ�����	�	��	���!�"�#�'�"��������������������|���������������������������H�H�>�<�?�H�N�U�X�\�W�U�H�H�H�H�H�H�H�H�O�I�B�6�0�)�$�(�)�.�6�B�O�V�Q�Q�T�U�O�O������'�4�@�G�@�;�4�'��������Y�P�O�H�>�G�M�Y�f���������������r�f�Y�������������ûлһлû��������������������������������������������������������������������������������������������������"�� �"�.�;�G�I�G�;�:�.�"�"�"�"�"�"�"�"�ʾʾ����������������ʾվ׾����׾ʾ�ìêäêìù��������ùùìììììììì�)��������)�B�O�[�h�n�l�{�u�j�\�B�6�)�x�[�R�l²��������������¿¦�m�m�m�n�z�������������z�m�m�m�m�m�m�m�m������ĿĸıįĳĿ�����
��,�*�+�
�������#����
�	�
���#�%�/�/�7�<�D�<�6�/�#�H�G�D�A�F�H�K�U�V�V�U�K�H�H�H�H�H�H�H�H�=�3�/�1�A�M�Z�s������������������s�Z�=àÚÓÍÇÂÀÁ�ÇÌÓÜâæìîðìà�	�� � �	������	�	�	�	�	�	�	�	�	�	��ƳƟƙƠƚƑƧƳ���������"�!��������b�Y�b�f�n�{Ňņ�{�n�b�b�b�b�b�b�b�b�b�bŔňŔřŠŭŹ����������������ŹŭŠŔŔ�����޾������	��	�	����������ľƾ־۾�	��.�Q�Z�Z�X�H�;�-�"�	��������������������������������������������ā�{�x�t�h�[�P�N�U�[�h�tāČĎĐčČČā�����������������������������������z�{����}�����������������������������������������'�1�2�.�/�'��������L�L�L�L�N�V�Y�e�q�r�~����~�~�r�e�a�Y�L�������%�#�(�5�5�A�A�>�7�5�(���D�D�D�D�D�D�D�D�D�EEE*ECEPEVENE*ED�D��C�>�7�<�C�O�\�c�]�\�O�M�C�C�C�C�C�C�C�C�b�[�]�b�n�u�{�{�{�n�b�b�b�b�b�b�b�b�b�bE�E�E�E�E�E�E�E�E�E�FFF$F0F.F$FFFE��;�7�/�"��"�(�+�/�5�;�H�T�Y�[�Z�V�O�H�;�I�E�=�;�0�/�0�=�D�I�N�V�]�Z�V�U�I�I�I�I�ݿѿĿ������Ŀſѿݿ������
�������ݻ:�*�'�&�-�:�_�l�������������������l�F�:������������������������Z�Y�N�C�B�G�N�U�Z�^�g�s�z�������w�s�g�Z�������������Ŀѿѿݿ�޿ݿؿտѿϿĿ����ܹ۹ϹʹϹӹܹ�������������ܹ�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E��e�\�L�G�L�S�_�e�~�������������������~�e�f�e�Y�M�A�G�M�Y�f�r�����z�r�g�f�f�f�f����
�
����$�%�$���������������$�0�4�9�0�.�$���������0�,�#� �"�#�'�0�<�>�E�<�0�0�0�0�0�0�0�0��������������������������ĳıĭĩĭĳĽĿ������������������Ŀĳĳ�����������Ľн���������ݽн�������Ç�z�t�r�t�s�n�l�n�ÒÓÞàáàÜØÓÇ���������޻������'�/�2�/�'�"����������������ûûлٻлû����������������U�P�H�<�;�9�<�E�H�U�a�n�q�n�n�d�a�V�U�U�G�?�:�5�:�G�S�U�^�S�G�G�G�G�G�G�G�G�G�G P N ; I - m i p B D T L X U ] 9 } g f I p R D F / R : w : { T ' G o ( 4 G L � V C A ` { 3 < J S : : % F  v W L 0 2 < 7 h _ h | l # i = ]    �  �  '  o  �  }  B  �  ?    o  l  �  W  �  t  �  `  r  {  �  �  �  6  {  �    �  �  j  h    h  ^  �  X  �  �      :      �  �    z  �  �  �  N  �  L  6  �  4  �  �  '  �  L  �  C  k  �  �  �    3=P�`<�C�;o���������o��h��1����}󶼋C����o�t��o�m�h������/��`B��h���o�}����#�
�m�h�@��#�
��9X�T����w���-�0 ŽH�9�T�����ixս����Y���t���C���%��7L��`B��+��%�Ƨ�hs��O߽��w�����hs��^5�������罣�
�J�� Ž��T��^5��{��9X�\���ͽ�xվ\)����1'��FB.B��B�|B&\�B&�B$m�BB��ByVB+B0��B,B ��B��B(+B�B�B�]B2��Bs�B �<B�B��B
�?BiB>;B�B��B��B��B�BrB��Be�B]UB	�}BB{�B	�B��B��B!'#B:�BB�XB�B��B�dB
��BB%�B �B��BBy�BC�B��B��B��B@_B@HB	�B	�B�KB��B�)B�'B
�2B�B=sB�BE�B&�CB%STB$��B=�B�BB�B+=	B0��B�`B �gB0wB7KB �BN�B��B2��B�B ��B�B@BBM'B<�B ��BđB:�B��B>"B+9A���B	B�5B@ B	��B0�B�B	�B��B��B!?�B>YB�B��B��B?�B�7B
�kB��B%�KB
B��B�WB��B<�BUB��B�iB@$B� B	�B	-�B@gB�B�(B��B
�EBȱA�׆A�J6A��SA�K'A(�T@��A���@A�UA��mAp�}A]��@���A��Aز�@�n@��@��A!��AK�*Aat�AQxA�=2A�wA���A��A�5�A���A�<}A?3=A���A[2B�A���A�V AXJ�A\�yA��A���A�l�A��?�+�?琥A��C�\lBA�>YC���A��XB2�A}x@��yA�I�A��Az:)?�[C��@	+;@��B	-�B	��A�A��A��A)^\A�S�@���@��pAŘkAS�A�v<A�ExA��TA�7VA%W@�n�A�xo@C�*A�~�Ai9�A] *@�	XAą�A�:@��M@�Q@��pA!8LAL�_Aa�AQ�A�;�A��A��KA���A���A�wlAĊ�A=~A��A[թB��A��A�I]AX��A[5A��4A۪}A�T�A�H�?��?�x�A�roC��$B>�A�sC��A�w�B@wA|�@��A��?A��=Ax��?1zIC��@yU@ܔ�B	@{B	��A�vJA��FA�qgA'�Aɳ@�.@��ZA�U�AP�            N   i            	   3                  &                         ?               5         (            @                         :         &      	                	   
      5                           -                     A   =      -         ;                  !                     !   5               !         /            /            '            #                     '                  !                     !                           =   3      %         -                                       !   )                        '                        '            #                     #                                       !               NU��N��N ��P��oPV~fNo�O��mNif�N0�AP$��N=vO�JN99�O��N�?DOpJN"\�N#WN?gNC��N�pNN|�OO���P7�Na�O�N�7�N2On�UN��N&~�P2�ENMH[N��.N��O��N�J@N���N6��O��O�N�t�Nы-O�]~N�GWM�QN�`O3Z�N��OT8TO�>�N��Ow�N�^	N�fONcGO��&N��%M�:�NX��N#�CNd��N��O`�N��-O<)N L�N�N�i    \  �  �  v    �  �    &  �  /  �  �  �  �  �  �  �  �  �    �  �  �  �    �  '  �  p  F  �  3  �    N  P  2  ]    0  �  
�  �  �  	Q  _  &    �  G      �  r  �  �  �  %  �  n  �  '  �  	`  �  T  �=m�h<�9X;��
��`B��1��o��`B�o�49X��j�T����o���
���
��1��/��j�ě��ě����ͼ��ͼ������0 ŽC��\)�\)�\)�D����P��P��w��w��w�#�
��O߽<j�ixսL�ͽH�9�L�ͽL�ͽT���aG��q���q����+�u�y�#��o�����o��t���hs���P�������㽙�����-��������置{��{�ȴ9���`�ȴ9��/��`B#/<?=<0/(#jnvz�����������znkjj���������������������#<b{����{`I<0
����<Ubijg^XSI< ����
#&#
�������#5B[gx~gYNB)(),)(����������������������������������������068CGOXONC6-00000000ehity�����������xthe��������������������
#/3<HFA</#
MO[gd`_b[TOLJIMMMMMMst����������������ys��������������������pt������wvtpppppppp`hu����uha``````````�����������������������������������������������������������������������������L[t������������NHEFL()05>B95)'#$((((((((uz���������������wsuDHJU]abnu|znma[UMHDD_anz���znha________ #&/<HUY``]UH</#  ��������������EO[hihc[OIEEEEEEEEEEfmz�������������mbaf����������������������������������������������������GP[gt��������tg[NHEG�����

������������������������������qtx|�����}trmnqqqqqq
)O[hlgYWSOB:0

R[ahntv����th[QOQSRR������������������������������������������#)"#''"
���������������������������������������������"#$,/:<>FHJIFB</*##"��������������������rt~�����������ttrrrr������
 �������#0@MS]YTI0
�������������������������

���������qz�������������|zoqqmnuz���������zxrnnmmagnqnaUUTUaaaaaaaaaa������������������������� �������������������������������������������������������������������dgst������|tgeddddddLN[gtztqgb[NDLLLLLL�
 ##� �����)5BKDB=5*)(����������������������������������������stt~������������{tss��������������������ŭũŦŧŭŹŻ��������Źŭŭŭŭŭŭŭŭ�������������������������������������������������������������������������������������g�W�G�?�A�K�g������������������׽����{�v�z����������(�5�4����ݽĽ��������������������������������������������Z�N�A�4�'����(�5�U�g�{���������s�g�Z�ֺ˺ʺֺغ��������ֺֺֺֺֺֺֺ����������������������������������������`�G�<�9�>�G�T�`�����ĿͿ���ѿ����y�`��	�	��	���!�"�#�'�"��������������������|���������������������������U�M�H�?�=�F�H�I�U�W�[�V�U�U�U�U�U�U�U�U�O�I�B�6�0�)�$�(�)�.�6�B�O�V�Q�Q�T�U�O�O������'�4�@�G�@�;�4�'��������Y�X�R�N�L�I�D�L�Y�f�������������r�f�Y�������������ûлһлû��������������������������������������������������������������������������������������������������"�� �"�.�;�G�I�G�;�:�.�"�"�"�"�"�"�"�"�ʾʾ����������������ʾվ׾����׾ʾ�ìêäêìù��������ùùìììììììì�)��������)�B�O�[�h�n�l�{�u�j�\�B�6�)�m�i�o¦¿��������������¿¦�m�m�m�n�z�������������z�m�m�m�m�m�m�m�m������ĿĸıįĳĿ�����
��,�*�+�
�������#����
�	�
���#�%�/�/�7�<�D�<�6�/�#�H�G�D�A�F�H�K�U�V�V�U�K�H�H�H�H�H�H�H�H�M�F�A�9�3�5�=�A�M�Z�f�r�|�����v�s�f�Z�MàÚÓÍÇÂÀÁ�ÇÌÓÜâæìîðìà�	�� � �	������	�	�	�	�	�	�	�	�	�	ƳƨƠƛƣƟƳ������������ �������Ƴ�b�Y�b�f�n�{Ňņ�{�n�b�b�b�b�b�b�b�b�b�bŔňŔřŠŭŹ����������������ŹŭŠŔŔ�����޾������	��	�	������������������	��"�.�6�<�<�7�1���	�������������������������������������������h�d�[�V�U�[�[�h�t�tāĆĈāā�t�h�h�h�h����������������
������������������������z�{����}�����������������������������������������'�1�2�.�/�'��������L�L�L�L�N�V�Y�e�q�r�~����~�~�r�e�a�Y�L�������%�#�(�5�5�A�A�>�7�5�(���D�D�D�D�D�D�D�D�D�EEE*ECEPEVENE*ED�D��C�>�7�<�C�O�\�c�]�\�O�M�C�C�C�C�C�C�C�C�b�[�]�b�n�u�{�{�{�n�b�b�b�b�b�b�b�b�b�bFFE�E�E�E�E�E�E�E�E�FFF$F*F+F$FFF�;�7�/�"��"�(�+�/�5�;�H�T�Y�[�Z�V�O�H�;�I�E�=�;�0�/�0�=�D�I�N�V�]�Z�V�U�I�I�I�I�ݿѿĿ¿����Ŀſǿѿݿ������������ݻ:�,�(�(�-�:�F�l�x���������������x�l�F�:������������������������Z�N�D�B�H�N�W�Z�]�g�s�y���������v�s�g�Z�������������Ŀѿѿݿ�޿ݿؿտѿϿĿ����ܹ۹ϹʹϹӹܹ�������������ܹ�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E��e�]�M�M�T�_�e�~���������������������~�e�f�e�Y�M�A�G�M�Y�f�r�����z�r�g�f�f�f�f����
�
����$�%�$���������������$�0�1�7�0�)�$���������0�,�#� �"�#�'�0�<�>�E�<�0�0�0�0�0�0�0�0��������������������������ĳĲĮīįĳĿ������������Ŀĳĳĳĳĳĳ�����������Ľн���������ݽн�������Ç�~�z�u�s�v�w�zÇÊÓÚàààÚ×ÓÇÇ���������������!�'�+�/�,�'������������������ûϻû��������������������U�P�H�<�;�9�<�E�H�U�a�n�q�n�n�d�a�V�U�U�G�?�:�5�:�G�S�U�^�S�G�G�G�G�G�G�G�G�G�G P N ; G 9 m d 2 B N T L \ U ] 9 } g f I p R D H / R : w  { T * G o ( + E & t V C A ` { 3 < 8 S : 8  F   v W L . 2 < & h _ ` | U  ] = ]    �  �  '  �  �  }  �  u  ?    o  l  �  W  �  �  �  `  r  {  �  �  �  4  {  �    �  �  j  h    h  ^  �  O  �  �  �    :      �  �      �  �  �    �  ?  6  �  4  �  �  '  l  L  �  �  k  E  �  ]    3  Dr  Dr  Dr  Dr  Dr  Dr  Dr  Dr  Dr  Dr  Dr  Dr  Dr  Dr  Dr  Dr  Dr  Dr  Dr  Dr  Dr  Dr  Dr  Dr  Dr  Dr  Dr  Dr  Dr  Dr  Dr  Dr  Dr  Dr  Dr  Dr  Dr  Dr  Dr  Dr  Dr  Dr  Dr  Dr  Dr  Dr  Dr  Dr  Dr  Dr  Dr  Dr  Dr  Dr  Dr  Dr  Dr  Dr  Dr  Dr  Dr  Dr  Dr  Dr  Dr  Dr  Dr  Dr  Dr      �  �  �  �  �  �  h  G  &    �  �  �  �  �  �  �  {  \  R  H  >  3  &      �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �    *  T  �  �  �  {  F    �  }  ;    �  �  f    �  C  �  %  =  �    S  o  v  l  X  C  A  G  1    �  �  �    o  �  �  `      	    �  �  �  �  �  �  �  �  �  �  �  k  S  ;  #    �  �  �  �  �  �  �  |  n  X  8    �  �  j    �  K  �  �  |  �  �  �  �  �  �  �  z  Q  #  �  �  �  O    �  �  �  R          �  �  �  �  �  �  �  �  �  �  x  P  '  �  �  �        �  #  !      "           �  �  �  E  �  &  a  �  �  �  �  �  �  �  �  �  x  j  \  N  A  3     �   �   �   �  /  /  )      �  �  �  �  �  y  R  (  �  �  �  �  �  |  �  �  �  �  �  �  �  �  �  �  q  U  6    �  �  �  f  >    �  �  �  �  �  �  �  �  �  �  g  E    �  �  �  i  2  �  �  /  �  �  �  �  �  k  o  y  |  {  u  c  J    �  �    �    �  �  �  �  �  �  �  �  q  S  2    �  �  �  \  $  �  u  �  8  �  �  �  �  �  �  �  �  }  ]  >    �  �  �  �  �  �  r  \  �  �  �  �  �  �  �  �  �  �  �    q  c  Y  N  D  9  /  $  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  |  s  h  ]  R  H  6  "    �  �  �  y  o  e  Z  O  C  6  '      �  �  �  �  �  �  �  �  �              �  �  �  �  �  �  s  S  2    �  �  �  �  �  �  �  �  �  �  g  :  �  �  Z  �  �  )  5    �  �  E  �    [  �  �  �  �  m  :    �        �  �    Y  �    �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �    �  �  �  n  N  )    �  �  a  "  �  �  �  r  K    �  �  �      �  �  �  �  �  �  �  �  �  y  d  N  7  	  �  ?  �  C  �  �  |  w  q  o  l  i  Z  8    �  �  �  �  �  f  H  *      l  �    &  !      �  �  �  _    �  p  �  4  W  w  +  �  �  �  w  c  ?    �  �  J  I    �  �  I    �  �  �  |  p  o  o  o  o  o  o  o  o  o  o  n  n  n  n  m  m  m  l  l  �  0    �  �  �  f  /  �  �  U    �  Z  �  �  n  }  !  >  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  3  #      �  �  �  �  �  �  �  �    L  �  �  ?  �  �  8  �  �  ~  q  Z  ?    �  �  �  |  M    �  �  �  N    �  T  p  �  �  �  �  �  �  �  �     �  �  �  �  `    �    R  �  L  M  M  K  B  6  &    �  �  �  �  m  I  '    �  �  y  =  �  �  }  �  #  D  N  F  :  "  �  �  �  G  �  �  h    �  �    "  &  *  .  1  .  +  (  %          �  �  �  �  �  �  ]  N  E  9    �  �  �  �  �  _  Y  @  %  �  �  k    �  `            �  �  �  �  �  �  �  �  o  D    �  �  S  �  0  "    �  �  �  �    \  8    �  �  �  }  Z  8    �  �  �  �  m  U  R  x  d  K  2      �  �  �  d     �  ;  r  �  
�  
�  
�  
�  
�  
�  
�  
�  
  	j  �  �  	  }    y  �  �  �  c  �  �  �  �  �  �  �  �  �  |  o  c  Y  O  A  0    
  �  �  �  �  �  �  �  ~  r  f  Z  N  A  4  (      �  �  �  Q     �  	G  	N  	Q  	B  	  �  �  f    �  q    �  Q  �  ;  �  �  _  _  P  B  3  (      �  �  �  t  J     �  �  �  n  I     �  &    	  �  �  �  �  t  N  '  �  �  �  x  K     �   �   �   i  {  ~  v  k  ^  L  8  !    �  �  �  N    �  �  O    �  �  �  �  �  �  �  �  �  �  �  �  �  g  =  	  �  �  �  `    �  G  7  (      �  �  �  �  �  �  }  _  A  "    �  �  �  �      �  �  �  �  �  �  �  �  d  >    �  �  \    �  8  �    �  �  �  �  �  �  q  S  3    �  �  �  �  �  �  �  �  f  �  }  u  j  ^  O  >  ,      �  �  �  �  �  i  ?    �  �  r  h  ^  T  I  8  (      �  �  �  �  x  ]  C    �  �  |  �  �  �  �  i  2  �  �  �  R    �  �  N  �  s  �  P  �  �  �  �  �  �  �  �  x  ^  C  &  	  �  �  �  ~  P    �  �  v  �  �  |  m  _  N  8  #    �  �  �  �  �  }  a  F  *     �  �      (  0  .  %    �  �  �  �  q  @  
  �  }  ,  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  ~  o  a  S  E  n  a  S  F  7  "    �  �    )  C  M  K  I  H  1    �  �  �  �  �  �  �  s  G    �  �  �  �  `  9    �  �  X    �  '      �  �  �  �  �  �  �  �  �  �  �  s  E    �  �  t  �  �  �  �  �  �  �  �  v  ]  =    �  �  z  .  �  f  �  �  	  	:  	V  	X  	;  	  �  �  e    �  a    �  n    �  �  Y  d  �  �  �  �  �  �  �  �  �  u  X  6         3  o  �    \  T  N  =  &  �  �  �  g  .  �  �  h  $  �  U  �  F  �    "  �  �  �  �  v  e  T  D  3  $      �  �  �  �  
  >  ~  �