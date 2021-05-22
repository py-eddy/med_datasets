CDF       
      obs    F   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type         ����         	amplitude                   comment       �Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM���   
add_offset               min       ?h�t�j~�   max       ?�p��
=q       �   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       M�(�   max       P\��       �   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       ��^5   max       <��
       �   effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @?^�Q�   max       @Fz�G�{     
�   �   effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ���
=p�    max       @v���R     
�  +�   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @0         max       @Q�           �  6�   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @̶        max       @�'            7`   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       �O�   max       ;D��       8x   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A��"   max       B2F^       9�   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A�~r   max       B2p       :�   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       >mL�   max       C��x       ;�   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       >C/P   max       C��b       <�   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max          t       =�   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min             max          5       ?   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min             max          /       @    
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       M�(�   max       PA��       A8   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6��C-   
add_offset               min       ?�_ح��V   max       ?�>BZ�c        BP   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       ��^5   max       <u       Ch   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @?fffffg   max       @Fz�G�{     
�  D�   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ?�z�G�    max       @v���R     
�  Op   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @.         max       @Q�           �  Z`   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @̶        max       @�@           Z�   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         @�   max         @�       \   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6��C-   
add_offset               min       ?�-�qv   max       ?�>BZ�c      �  ]   s               	      	                            	                     +               /         5         
   ,                        '   8   
   +         I               D            &   	      $         9      
   ?      PV�8N�n�OvSO2�WNۗ�N�$O��+O	��N��O.#�PбOu�O[�P,�;NKN�mO��N~O1˼OICKNHi0NKQNm��P\��PP�O�;�O"�O��OO$0�N�-4O��cO��M�(�N(�O%��O���N���N���NV�#O�u�Nu|Np0O
H3O�JEO��UO$��P'F�O���N���P��N��O���O�,]N�'�P<��Oe�OJ�NL�tO�_O��OB�P%POf�\OXe<P(�2N��ZNV]�O�/N�j�N�q<��
<���<T��<t�;��
�ě��#�
�#�
�#�
�49X�49X�D���T���e`B�u�u�u�u��o��o��o��o��o��o��C���C���C���C���t����㼣�
���
���
��1��j�ě����ͼ�/��`B��h�������+�+�C��C��\)��w�',1�8Q�<j�<j�<j�D���P�`�e`B�ixսixսm�h�m�h�m�h�u�����hs��hs���㽝�-��^5�����$(' ������������������������������������������������������������������������������������������������������������h\@.'(/6CCIhu�����lh������

 ������36<ABO[[]_^[SOB62533����������"*5BNesqhN5)��������������������������������������������8EB��������"#-/276///#"""""""""(/4;<?<<</-,((((((((chot����������{thf_cqt��������ttqqqqqqqqQTakmsxz|�{ymaTPMMQy��������������xpquy������������������������

������������������������������#Ibn{������{U<0
KRamz������zmaTHABFKYg�������������t[VSY57BOT[ghlth[OFB62005r�����������{nnqroor�����


���������Y[_gtz�����tg[[XYYYY`hnqsz�����������na`�� ).01/)����$)-66:6)%!$$$$$$$$$$������������������������!���������&),-065)�����:<BHSU[ahlaUHA<5::::NOPZ\huz����uh\ONNNNAIUbhmeb]UIEAAAAAAAA�������������������������������������������������������������������������������W\h�����������h\WUUW�����������������������
#/5//)#
�����B@AJLUanz�����zna<Begt������������oka`e�����
�������������#/5;ELUbgmH</#���������������������������������������9=JdgmmonqnpobNB5029ot~�������������ztoo�������� 	��������X[gt��������ztjg`[[X����������������������������������������Y]l������������tb\YY��������������������[]dht�vrha[UOKIOWX[��)/7BLE6������#0<ITVWUQLID<#��������
������������������������������vtrtvz��������������������������������y���������������omny $)5<=85-)z{��zvna_VURPQU`anzz�A�5����)�5�N�g�s���������������s�Z�A�n�l�g�a�U�N�Q�T�U�]�a�c�n�n�q�q�q�o�n�n�
�������������
��#�/�<�H�Q�H�E�<�/��
�g�^�c�g�i�l�l�m�s�w�����������������s�g���������������!�-�9�:�>�:�0�!����ݽнĽýĽ˽нݽ�������������z�r�T�;�.�"�����"�0�;�A�7�;�T�U�m�z����������üú����������������������m�i�`�T�R�L�S�T�`�m�y�������������y�m�m����������������������� ���	�����������g�Z�N�A�8�<�H�g���������������������s�g��������}�t����������������������������	� �����߾Ӿ׾����	��"�'�)����	�����"�;�N�y�������������y�`�G�;�.��H�D�<�7�<�H�U�a�a�a�W�U�H�H�H�H�H�H�H�H�
��
���#�/�/�/�+�#��
�
�
�
�
�
�
�
����¿¶³²²¿������������������������ìåæêìù������ûùùììììììììƳƬơƧƩƳ��������������������������Ƴ���!�)�4�=�B�[�h�tĀ�z�t�[�O�B�6�)�������������������������������������������t�r�g�f�g�h�k�t�z�x�w�t�t�t�t�t�t�t�t�tÓÇÇ�z�o�q�zÇÓÔÚÞÓÓÓÓÓÓÓÓ�����t�S�C�>�A�K�`�����������������������	���������	�"�/�H�T�a�w�u�x�w�p�T�H�;�	�������������������Ŀѿۿ߿�ۿӿ��������H�H�J�P�U�[�a�c�n�zÂÆÄ�z�w�p�n�a�U�H�����������Ŀѿ����)�4�2�(���ݿĿ���D�D�D�D�D�D�D�D�D�EEEEEEEED�D�D߾���߾߾�����	����	�	������������������������������������*� ��s�f�T�M�I�C�M�S�Z�f��������ľ�������s���������������������������������������������������������������������������������m�h�e�d�b�f�m�v�z�~���������������z�v�m�[�O�6�.�B�J�h�tāĚĦĬĮĩĝčā�t�h�[����������������������������������꾾�������������������������¾¾����������-�(�)�*�-�:�F�P�F�A�?�:�-�-�-�-�-�-�-�-�����������Ƴ����������$�0�4�6�4�0�$���ݼ����� ����������������ìëåìñù����������ùìììììììì���������������������������	��	������������~���������л�������Իû��������������4�@�M�Y�f�t�y�w�i�Y�M�4���������������������������������������������a�H�;���������������/�H�T�a�i�������a��������������)�8�B�O�Q�N�<�)������ܻջջܻ޻�����������ܻܻܻܻܻ�E�E�E�E�E�E�E�E�E�FF$F=FHFHF=F.FE�E�Eͽ�
�������!�&�(�.�2�:�<�:�.�,�!�ǈ�o�b�V�K�N�O�H�E�>�I�V�oǈǡǫǭǡǔǈ�������������������*�6�C�O�T�N�C�6�������ŹŷŶŭŬūŭŹ�������������������ƻл����������߼�'�@�Y�g�h�b�X�6������ìéäàæìõù������������������ùòì�Y�L�E�D�@�=�<�H�L�Y�r��������s�v�r�e�YFJF@F=F3F3F=F?FJFVF\F\FVFJFJFJFJFJFJFJFJĿĳĦĠĝĘĜĳ�������������������ĿĿĹķĿĿ�������������������������ĿĿ�����������ùϹܹ����ܹչϹù����������l�N�I�U�`�������Ľӽ������ݽн����������������������½н����ݽνĽ������������������Ǽּ����������ּʼ����~�r�Y�H�F�M�Y�r�������������ɺ����~�a�a�T�R�H�>�;�/�(�$�$�/�;�H�T�W�Z�a�a�a�ĿÿĿʿѿݿ����ݿѿĿĿĿĿĿĿĿ������������#�/�<�C�H�U�V�W�Q�H�<�/�ÇÇ�z�n�i�g�n�zÇÌÓ×ÓÉÇÇÇÇÇÇ�������ĿͿѿտӿѿĿ������������������� 8 p t O Q 6 k N 6 A   H + Y I k " d A n V � K A 8 C Y C 4 # V < Q P e J G k k Q C 3 U H < ] N K B [ X o N U i + Q K $ J 7 F 6 W Q " M   C >  �  '  �  �    �  �  :    �  �  8  �  X  7  \  J  f  �  �  �  �  �  �  �  �  ]  W  k  	  K  �  &  a  �  �  �  	  �  �  8  z  M  �  �  �  �  �  �  �  7  �  �    !  C  �  a    ;  f  &  �  �  K    �  i  �  	�\�49X;D���o��o�e`B��1��t���o��j�0 ż�����49X��1��t��ě�����o��h�ě�����ě��q���D����P���8Q콅���h�C������j���ͽo��7L�+�C����Y���P�49X�<j��t���E��0 Ž���T���<j��l��y�#��t��}�L�ͽ�xս�����7L��+�\��+�� Ž�vɽ��㽙�����#��E����T�O߽����l�B�GB&�B��BY�B,eLB)ۃB1�B\�B�LBILB��B	,B�+B;=B8�BQMB��B�]A��"B O�B�
B|�B �1B'/MA�N�B
�B�IB)��B�$B	��B�xBO�Ba�B��B1LB7B��B2F^B';�B��B�=B�gB�UB��B �B��BwB
�B#L�B��B8�BEBSB
��B0B	�aB!�kB6�B
�-B�eB�Bx�B&B-<�B/8B%�BQ�B�B��B�[BɺB��B�BAB,?vB)�B1�QB��B�-B@BB��B6�B�$B�B?B@SBIDB�dA�~rB � B�vBDpB ĄB' �A�_5B
;kB�$B)�2BA�B	�B�bB?�BA+B�BUSB�B�UB2pB'LoB�;B��B��B�HB)�B �LB�B��B
�B#?�B@�B�uB?�B�=B
� B1B	�MB!�'B@fB
@?B��BHYB�pB%��B-%�B�~B2B�;B
��B�WBXoA��GA�
}A��A��@d�:A+�Ab�HA��Aj��A��A��,AI�AY��Ag��A��TA��A��A�3NBcA�,�A���A���AɌ5A�O�A���Av�A� A~�-C�D�AX�+A���AEϻA��A��A��BA�E�A�?�ANJ�@yC�B��A�A͓AA��@�bk@�M�A��!A��A��@�o�C���A]�B3HA��+A��@���A��z?�eC��xA��A��v>mL�A"��A%�A�@
�A��A{�A�#A���Av�<A���A��A�1�A���@m�A+�Afa�AіNAj�pA��tA���AH�AZ�2AkэA�|MA�sZA�E�À�B�Aڒ,A�s�A�SAȇA��fA���AvgA�fIA	�C�G�AX�/A��AE+A��KA��uA���A��bA�}~AM/@u�JB	(�A��A�|aA�y�@��"@�A�U�A���A��e@��PC���A� B8�A��+A�l�@���A͝(?���C��bA�vA��>C/PA2pA# �A�{@��A���A|�rA�c�A���Av�    t               	      	         !                  
            	      	   ,                0         6         
   ,   	                     (   9   
   +         I               E            '   
      $         :         ?         /                  (            '         2                              3   '         '         #               !            !            #         -   #      )      !   !      5            %         /         /                  !                  (                     )                              /   %                                                         #         +         )         !      /                     /         )               O�9�N_rgOvSO2�WNۗ�N]TO��+O	��N�'�N�$�O�c6N�;�O)+�PnNKN�mN��1N~O[�OICKNHi0NKQNm��PA��O�OO�G�N���O��O:�Nϧ�Oy1OO�M�(�N(�N�`*O��NX$iN���NV�#O�^kNu|N:��O
H3O�JEOMLO	�Pf3O��N���O�W$N�d�O�gBO�,]N�'�P*K�N��UOJ�NL�tOϗN��=Nv��P�Of�\OXe<P�MN��ZNV]�O���N�j�N�q  N  
�  %  �  �  �  R    �  0  �    �  �  �  X  �    �    �  �  z  �  �  �  B  �  
�  ^  {  �  �  �  �  �  h    w  �  1  �  ,  �  �  0  �  �  T  [  �  �      |  �  M  R  �  �  �  L  �  �  �  �  w  U  j  ���<u<T��<t�;��
��`B�#�
�#�
�49X��o�ě��T����o����u�u����u��t���o��o��o��o���
���
���
������/��9X���
��1�C����
��1���ͼ�h��/��/��`B�o���o���+�L�ͽ\)��P�t���w�L�ͽ0 ŽD���<j�<j�L�ͽm�h�P�`�e`B�u�m�h��O߽q���m�h�u��\)��hs��hs���罝�-��^5�����

������������������������������������������������������������������������������������������������������������h\@.'(/6CCIhu�����lh������

 ������?BDOW[\^][OB97??????���
���������)5BNR[^d`[NB5/) ������������������������������������������������4;)����"#-/276///#"""""""""(/4;<?<<</-,((((((((fhhty��������tphffffqt��������ttqqqqqqqqRT]afmqvxyuma_TRNNRRy��������������xpquy������������������������

������������������������������#<Ubn������{b<#DIJMTamz�����wmaTHDDV[`g������������ti\V?BEOX[bd[OLB>8??????{{�����������}{xwx{�������	


��������Z[dgtw�����tg\[YZZZZaainrtz�����������na�()++*'#����$)-66:6)%!$$$$$$$$$$�����������������������������������
"'++./(����AHUYaba^UNH<AAAAAAAANOPZ\huz����uh\ONNNNAIUbhmeb]UIEAAAAAAAA�������������������������������������������������������������������������������W\h�����������h\WUUW������������������������
#&)'#
������LNUanz�����ynaUEFBBLgt�������������plbbg�����
�������������#/7FQ^bd[H</#����������������������������������������9=JdgmmonqnpobNB5029ot~�������������ztoo�����������������egkt�����trigeeeeee����������������������������������������`gq�����������tc^[[`��������������������NOZ[hktwtsh[SONNNNNN����).6EB,�����#0<ITVWUQLID<#��������
������������������������������vtrtvz��������������������������������q{��������������vtpq $)5<=85-)z{��zvna_VURPQU`anzz�g�Z�H�@�@�G�N�Z�g�������������������s�g�n�m�a�V�V�V�_�a�b�m�n�o�n�n�n�n�n�n�n�n�
�������������
��#�/�<�H�Q�H�E�<�/��
�g�^�c�g�i�l�l�m�s�w�����������������s�g���������������!�-�9�:�>�:�0�!����ݽнƽͽнݽ���������������z�r�T�;�.�"�����"�0�;�A�7�;�T�U�m�z����������üú����������������������`�W�T�O�T�^�`�m�y�}�����y�m�`�`�`�`�`�`����������������������������������������s�g�\�V�Z�a�g�o�s���������������������s������v��������������������������������������߾ؾ�����	��"�#�%���
��`�T�;�.����"�;�G�T�y�������������y�`�H�D�<�7�<�H�U�a�a�a�W�U�H�H�H�H�H�H�H�H�
��
���#�/�/�/�+�#��
�
�
�
�
�
�
�
��������¿¼¿��������������������������ìåæêìù������ûùùììììììììƳƮƧƤƧƬƳ����������������������ƳƳ���!�)�4�=�B�[�h�tĀ�z�t�[�O�B�6�)�������������������������������������������t�r�g�f�g�h�k�t�z�x�w�t�t�t�t�t�t�t�t�tÓÇÇ�z�o�q�zÇÓÔÚÞÓÓÓÓÓÓÓÓ���x�b�Q�K�J�T�h�������������������������H�;��	����������"�/�H�T�a�m�r�r�g�T�H���������������������Ŀѿֿۿ߿ؿпĿ����U�T�U�Y�a�e�n�z��~�z�p�n�a�U�U�U�U�U�U�ѿĿ��������ɿѿݿ����� ��������D�D�D�D�D�D�D�D�D�D�D�EEEEEED�D�D�����������	����	���������������������������������������)��f�`�V�Y�Z�f�i�s�������������������s�f���������������������������������������������������������������������������������m�i�h�i�d�h�m�z�������������}�z�m�m�m�m�h�[�I�B�:�8�B�O�h�tčĚĦĪĤėčā�t�h�����������������������������������꾾�������������������������¾¾����������-�(�)�*�-�:�F�P�F�A�?�:�-�-�-�-�-�-�-�-���������������$�)�0�2�4�2�0�$�������ݼ����� ����������������ùíìæìõù����������ùùùùùùùù���������������������������	��	������������~���������л�������Իû��������%����'�4�@�M�Y�f�j�q�n�f�]�Y�M�@�4�%�������������������������������������������������������/�H�T�a�������z�a�H�;���������������)�2�5�B�E�A�;�)������ܻջջܻ޻�����������ܻܻܻܻܻ�E�E�E�E�E�E�E�E�FF$F1F=FDFDF:F+FFE�E������	����!�%�'�.�1�:�.�+�!��ǔǈ�{�t�o�V�R�R�L�H�I�V�o�{ǉǞǪǪǡǔ�������������������*�6�C�O�T�N�C�6�������ŹŷŶŭŬūŭŹ�������������������ƻл��������л��'�@�Y�a�f�_�U�4������ùóìçììù��������������ùùùùùù�Y�L�E�D�@�=�<�H�L�Y�r��������s�v�r�e�YFJF@F=F3F3F=F?FJFVF\F\FVFJFJFJFJFJFJFJFJĳĬĦĢĠĜĢĳ������������������Ŀĳ������ĿĺĿ��������������������������عù¹����������ù̹ϹڹٹϹĹùùùùùý������l�O�J�V�`�����Ľݽ������ڽĽ��������������������½н����ݽνĽ������������������Ǽּ����������ּʼ������~�Y�L�P�Y�r�~�������ֺ������ɺ����a�a�T�R�H�>�;�/�(�$�$�/�;�H�T�W�Z�a�a�a�ĿÿĿʿѿݿ����ݿѿĿĿĿĿĿĿĿ��/������� �
���#�/�<�H�S�T�N�H�D�<�/ÇÇ�z�n�i�g�n�zÇÌÓ×ÓÉÇÇÇÇÇÇ�������ĿͿѿտӿѿĿ������������������� < w t O Q 9 k N . H " B ' W I k + d 0 n V � K ? 8 @ \ 4 1 ! X 3 Q P q F B k k @ C 5 U H , V K B B [ V r N U `  Q K ! V 7 H 6 W G " M  C >  y  �  �  �    |  �  :  �  {    �  h  �  7  \  �  f  ;  �  �  �  �  d  =  `  �  .  B  �  #  �  &  a  d  U  n  	  �    8  Z  M  �  �  ]  �  u  �  _    |  �    �  �  �  a  �  �  �    �  �  �    �  +  �  	  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  `  C  �  	z  
  
  
�  &  L  E  *  
�  
�  
[  	�  �  �  k  D  �  	H  	�  
j  �  �  �  /  
�  
  	v  �  1  �  �  ,  m  �  �    6  %        �  �  �  �  �  j  2  �  �  s  4  �  �    �  �  �  �  �  �  �  z  a  G  -    �  �  �  r  I  "  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  }  h  P  8  !       �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  R  G  8  !    �  �  �  �  �  �  �  �  x  j  b  [  U  R  Q    �  �  �  �  {  b  I  <  0  ,  .  3  M  g  q  u  s  h  ]  �  �  �  �  �  �  �  �  �  �    h  R  9        �   �   �   �  �  �  �  �  �      *  /  )      �  �  �  c  .  �  �  q  $  Q  m  }  �  �  �  �  �  �  �  �  �  i  +  �  m  �  �  &                    	      �  �  �  �  �  j  :  	  �  �  �  �  �  �  �  �  t  _  H  *    �  �  u  =     �  j  �  �  �  �  �  �  �  w  P  "  �  �  j    �  �  �  h  )  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  X  M  B  7  ,  !      �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  k  I  &   �   �    #  3  C  R  ]  X  S  O  J  C  9  0  &        �  �  �  �  �  �  �  �  �  �  �  p  W  ;    �  �  �  n    �    j    �  �  �  �  �  �  �  g  8  �  �  �  �  �  �  �  l  C    �  �  �  �  �  �  �  �  �  �  �  |  [  +  �  �  �  p  ?    �  �  �  �  �  �  �  �  �  �    #  9  I  N  R  W  \  `  e  z  m  `  P  ?  .      �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  y  F    �  �  T    �  �  L  �  �    �  �  �  �  �  �  g  B     	  �  �  �  �  Q    �  }    �  �  �  �  �  �  �  �  �  �  l  T  9    �  �  �  O    �  �  t  �  �  �  �  
  #  2  @  =  5  *    �  �  ]    �  �  �  �  �  �  �  �  �  �  �  �  �  �  z  Q  "  �  �  `    �   �  
V  
�  
�  
�  
w  
Z  
5  
  	�  	�  	5  �  ~  �  l  �  +  �  #  �  Y  [  ]  W  P  A  2  !    �  �  �  �  �  o  H    �  �  :  x  z  z  x  o  f  ^  T  D  1      �  �  �  �  `  "  �  �  �    K  p  �  �  �  {  _  =    �  �  e      �  m  �   O  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  |  s  j  �  �  �  {  l  Z  @  &    �  �  �  �  �  �  �  �  v  g  W  m  t  {  �  �  �  �  �  x  o  d  R  5        �  �  �  ?  �  �  �  �  �  �  �  �  g  ;    �  �  g    �  @  [  I  %  C  I  P  X  a  g  c  _  V  K  ;  '    �  �  �  r  G     �      �  �  �  �  �  �  �  �  f  C    �  �  �  e  4     �  w  r  l  g  b  \  W  Q  K  E  ?  9  3  ,  $          �  �  �  �  �  �  �  �  �  y  a  N  1    �  �  v  :  �  �  )  1    �  �  �  �  �  �  �  x  d  O  4    �  �  �  q  <    �  �  �  �  �  �  �  i  K  '  �  �  �  ^    �  �  F  �  �  ,  +  +  +  *  &       �  �  �  �  \  7    �  �  l    �  �  �  �  �  m  �  R  �  ~  d  7  �  �  D  �  �  f    �  �  s  $  Z  �  �  �  �  �  |  @  �  �  W  �  _  �  �  �  �  �     (  /  -  *         �  �  �  �  �  �  �  s  [  <  �  �  �  �  �  �  �  �  �  �  �  g  =  �  �  k    �  m    �  �  w  �  �  �  z  l  [  F  /    �  �  �  |  a  A    �  �  �  T  E  7  )        �  �  �  �  �  �  V  $     �  �  �  e    M  [  V  ;    �  �  0  
�  
h  
  	�  	2  �    &  q  t  G  �  �  �  �  �  �  �  �  �  t  I    �  �  x  C    �  O  �  �  �  �  �  �  �  l  G  (  �  �  �  y  y  $  �  �  K  �  �          �  �  �  �  �  �  w  [  V  8    �  �  �  X  >                  �  �  �  �  �  �  �  �  �  �  }  l  <  |  a  6        �  �  �  h  0  �  �  2  �  �      j  �    ;  d  �  �  �  �  �  �  v  K    �  �  d    �    n  M  4            #      �  �  �  �  �  �      )  8  R  V  Y  C  ,    �  �  �  �  �  f  B    �  �  �  �  U  &  �  �  �  �  �  �  j  I  )    �  �  �  P    �  A  �  !  �  �  �  �  �  �  �  �  �  �  s  d  W  K  =  .      �  �  �  j  �  2  `  �  �  �  �  �  �  �    T  (  �  �    -  >  V  F  K  ?  2  &        �  �  �  �  k  #  �  Y  �  �  �  �  �  �  �  �  �  �  ~  b  A    �  �  �  }  Y  '  �  ^   �   b  �  �  �  �  �  x  W  5    �  �  �  x  I    �  �  �  O    �  �  �  �  �  �  c  '  �  �  O    �  �  �  A  �  [  �  :  �  �  �  �  �  p  c  W  J  :     �  �  �  Z  
  �  R  �  i  w  s  n  Y  C  $    �  �  �  |  Y  4    �  �  �  x  �  �  6  I  U  J  A  *    
�  
�  
f  
  	�  	/  �    R  �  �  �  �  j  _  S  G  9  +      �  �  �  �  �  �  �  q  ]  T  K  B  �  U  (    �  �  �  }  M    �  �  �  U     �  �  p  ,  �